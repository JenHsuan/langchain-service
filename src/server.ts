import express from 'express';
import bodyParser from 'body-parser';
​import { HttpResponseOutputParser } from "langchain/output_parsers";
import { Document } from "@langchain/core/documents";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { createConversationalRetrievalChain, createDocumentRetrievalChain, createFinalRetrievalChain, createRephraseQuestionChain, initializeVectorstoreWithDocuments, loadAndSplitChunks } from './lib/helper.js';
import { pipeline } from "node:stream/promises";

//split the document
const splitDocs = await loadAndSplitChunks(
  './data/seanhsieh_cv_2024_04.pdf',
  1536,
  128,
);

console.log(splitDocs)

//vector store
const vectorstore = await initializeVectorstoreWithDocuments(splitDocs);
const retriever = vectorstore.asRetriever();

//model
const model = new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo-1106" });

//document retrieval chain
const convertDocsToString = (documents: Document[]): string => {
  return documents.map((document) => {
    return `${document.pageContent}\n`
  }).join("");
};

const documentRetrievalChain = createDocumentRetrievalChain(retriever, convertDocsToString);

//rephrase question chain
const REPHRASE_QUESTION_SYSTEM_TEMPLATE = 
  `Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.`;

const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human", 
    "Rephrase the following question as a standalone question:\n{question}"
  ],
]);
const rephraseQuestionChain = createRephraseQuestionChain(rephraseQuestionChainPrompt, model);

//answer generation chain prompt
const ANSWER_CHAIN_SYSTEM_TEMPLATE = `You are an interviewee, 
expert at coding and analyzing.
Using the provided resume, answer questions for the interviewing 
to the best of your ability using only the resources provided. 
Be verbose!
<context>
{context}
</context>
Now, answer this question using the above context:
{question}`;

const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human", 
    "Now, answer this question using the previous context and chat history:\n{standalone_question}"
  ]
]);

//conversational retrieval chain 
const conversationalRetrievalChain = createConversationalRetrievalChain(answerGenerationChainPrompt, documentRetrievalChain, rephraseQuestionChain, model);

//final retrieval chain
const httpResponseOutputParser = new HttpResponseOutputParser({
  contentType: "text/plain"
});
const messageHistories: {[key: string]: any} = {};

const getMessageHistoryForSession = (sessionId: string) => {
    if (messageHistories[sessionId] !== undefined) {
        return messageHistories[sessionId];
    } 
    const newChatSessionHistory = new ChatMessageHistory();
    messageHistories[sessionId] = newChatSessionHistory;
    return newChatSessionHistory;
};

const finalRetrievalChain = createFinalRetrievalChain(conversationalRetrievalChain, getMessageHistoryForSession).pipe(httpResponseOutputParser);

const app = express();
const port = 3000;
app.use(bodyParser.json());

app.get('/', (req, res) => {
  res.send('The server is working!');
});

app.post('/', async (req, res) => {
  const {question, sessionId} = req.body;
  if (question === undefined || sessionId === undefined) {
    return res.status(400).send("Missing question or sessionId");
  }

  const stream = await finalRetrievalChain.stream({
    question: question
  }, { configurable: { sessionId: sessionId } });
  
  await pipeline(stream, res);
});

app.listen(port, () => {
  console.log(`server is listening on ${port} !!!`);
});
