const decoder = new TextDecoder();

(async function(){
  // readChunks() reads from the provided reader and yields the results into an async iterable
  async function* readChunks(reader: any) {
    let result = '';
    let decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        return result;
      }
      console.log(value)
      result += decoder.decode(value, { stream: true });
      console.log(result)
      yield result;
    }
  }
  

const sleep = async () => {
  return new Promise((resolve) => setTimeout(resolve, 500));
}

  const response = await fetch(`http://localhost:3000`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify({
      question: "What's your strength?",
      sessionId: "1", // Should randomly generate/assign
    })
  });
  
  // response.body is a ReadableStream
  const reader = response.body?.getReader();
  
  console.log(reader)
  for await (const chunk of readChunks(reader)) {
    console.log("CHUNK:", JSON.stringify(chunk));
  }
  
  await sleep();

})();