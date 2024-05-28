import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever';
import { HumanMessage, AIMessage } from '@langchain/core/messages';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';

export async function conversationalRetrivalChainExample() {
  // 1. Build Retrival chain element
  // 1.1 Fetch document
  const loader = new CheerioWebBaseLoader('https://docs.smith.langchain.com/user_guide');
  const docs = await loader.load();

  // 1.2 Initialise splitter
  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments(docs);

  // 1.3 create vectorstore and prepare to be used for retrival
  const embeddings = new OpenAIEmbeddings();
  const vectorstore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
  const retriever = vectorstore.asRetriever();

  // 2. define history aware prompt for information retrival
  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder('chat_history'),
    ['user', '{input}'],
    [
      'user',
      'Given the above conversation, generate a search query to look up in order to get information relevant to the conversation',
    ],
  ]);

  // 3. Initialise model used for history aware retrival (Step 4) and interaction (Step 5)
  const chatModel = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4o',
  });

  // 4. create history aware retriever: based on the input, the retriever generates new documents from the source based on the new input topic (and all previous user input)
  const historyAwareRetrieverChain = await createHistoryAwareRetriever({
    llm: chatModel,
    retriever,
    rephrasePrompt: historyAwarePrompt,
  });
  // Example: Invoking this retriever with given history returns an array of adapted documents that will help the llm to answer correctly
  // const chatHistory = [new HumanMessage('Can LangSmith help test my LLM applications?'), new AIMessage('Yes!')];
  // const res = await historyAwareRetrieverChain.invoke({
  //   chat_history: chatHistory,
  //   input: 'Tell me how!',
  // });

  // 5. Create a chain that accepts an array of documents and passes them to a model
  // 5.1 Define Prompt template
  const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
    ['system', "Answer the user's questions based on the below context:\n\n{context}."],
    new MessagesPlaceholder('chat_history'),
    ['user', '{input}'],
  ]);
  // 5.2 Create the chain and add prompt
  const historyAwareCombineDocsChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt: historyAwareRetrievalPrompt,
  });

  // 6. Connect retriever and document accepting model chain to one chain
  const conversationalRetrievalChain = await createRetrievalChain({
    retriever: historyAwareRetrieverChain,
    combineDocsChain: historyAwareCombineDocsChain,
  });

  // 7. Test
  const result = await conversationalRetrievalChain.invoke({
    chat_history: [new HumanMessage('Can LangSmith help test my LLM applications?'), new AIMessage('Yes!')],
    input: 'tell me how."',
  });
  console.log(result.answer);

  // 8. Extend history
  const extended_chat_history = [
    new HumanMessage('Can LangSmith help test my LLM applications?'),
    new AIMessage('Yes!'),
    new HumanMessage('tell me how.'),
    new AIMessage(result.answer),
  ];

  // 9. Second call: Finish the conversation and check whether tool was called
  const result2 = await conversationalRetrievalChain.invoke({
    chat_history: extended_chat_history,
    input: 'Alright! Thank you and goodbye.',
  });
  console.log(result2);
}
