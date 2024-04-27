import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RunnableSequence } from "@langchain/core/runnables";
import { RunnablePassthrough } from "@langchain/core/runnables";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { 
    RecursiveCharacterTextSplitter
} from "langchain/text_splitter";
import { Document } from "@langchain/core/documents";

export const loadAndSplitChunks = async(pdfFile: string, chunkSize: number, chunkOverlap: number): Promise<Document<Record<string, any>>[]> => {
  //load the document
  const loader = new PDFLoader(pdfFile);
  const rawCS229Docs = await loader.load();
  
  //split the document
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: chunkSize,
    chunkOverlap: chunkOverlap,
  });

  const splitDocs = await splitter.splitDocuments(rawCS229Docs);
  return splitDocs;
}

export const initializeVectorstoreWithDocuments = async(splitDocs: Document<Record<string, any>>[]): Promise<MemoryVectorStore> => {
  //vector store
  const embeddings = new OpenAIEmbeddings();
  const vectorstore = new MemoryVectorStore(embeddings);

  //add chunks to the vector store
  await vectorstore.addDocuments(splitDocs);

  return vectorstore;
}

export const createDocumentRetrievalChain = (retriever: any, convertDocsToString: (documents: Document[]) => string): RunnableSequence<any, string> => {
  const documentRetrievalChain = RunnableSequence.from([
    (input) => input.question,
    retriever,
    convertDocsToString
  ]);

  return documentRetrievalChain;
}

export const createRephraseQuestionChain = (rephraseQuestionChainPrompt: ChatPromptTemplate<any, any>, model: ChatOpenAI): RunnableSequence<any, string> => {
  const rephraseQuestionChain = RunnableSequence.from([
    rephraseQuestionChainPrompt,
    model,
    new StringOutputParser(),
  ])

  return rephraseQuestionChain;
}

export const createConversationalRetrievalChain = (
  answerGenerationChainPrompt: ChatPromptTemplate<any, any>,
  documentRetrievalChain: RunnableSequence<any, string>,
  rephraseQuestionChain: RunnableSequence<any, string>,
  model: ChatOpenAI
): RunnableSequence<Record<string, unknown>, string> => {
  const conversationalRetrievalChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      standalone_question: rephraseQuestionChain,
    }),
    RunnablePassthrough.assign({
      context: documentRetrievalChain,
    }),
    answerGenerationChainPrompt,
    model,
    new StringOutputParser(),
  ]);

  return conversationalRetrievalChain;
}

export const createFinalRetrievalChain = (
  conversationalRetrievalChain: RunnableSequence<Record<string, unknown>, string>,
  getMessageHistory: any
): RunnableWithMessageHistory<Record<string, unknown>, string> => {
  const finalRetrievalChain = new RunnableWithMessageHistory({
    runnable: conversationalRetrievalChain,
    getMessageHistory: getMessageHistory,
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  });

  return finalRetrievalChain;
}