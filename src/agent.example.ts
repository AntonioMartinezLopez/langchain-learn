import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { createRetrieverTool } from 'langchain/tools/retriever';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { AgentExecutor, createOpenAIFunctionsAgent } from 'langchain/agents';
import { HumanMessage, AIMessage } from '@langchain/core/messages';
import { DynamicTool } from 'langchain/tools';
import { ChatMessageHistory } from 'langchain/stores/message/in_memory';
import { RunnableWithMessageHistory } from '@langchain/core/runnables';
import { ToolCallbackHandler } from './callbacks/toolCallbackHandler.js';
import { StreamCallbackHandler } from './callbacks/streamCallbackHandler.js';

export async function agentExample() {
  // 1. Create tools that the agent can select to use
  // 1.1 Tavily search engine for searching recent information, e.g. weather
  const searchTool = new TavilySearchResults({ apiKey: process.env.TAVILY_API_KEY });

  // 1.2 Retrival tool for getting information about lang chain
  const loader = new CheerioWebBaseLoader('https://docs.smith.langchain.com/user_guide');
  const docs = await loader.load();

  // 1.2.1 Initialise splitter
  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments(docs);

  // 1.2.2 create vectorstore and prepare to be used for retrival
  const embeddings = new OpenAIEmbeddings();
  const vectorstore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
  const retriever = vectorstore.asRetriever();

  // 1.2.3 Turn retriever into a tool
  const retrieverTool = createRetrieverTool(retriever, {
    name: 'langsmith_search',
    description: 'Search for information about LangSmith. For any questions about LangSmith, you must use this tool!',
  });

  // 1.3 A custom tool for recognizing conversation end
  const endConversationTool = new DynamicTool({
    name: 'end_conversation',
    description: 'To be called when a user ends the conversation or does not need any more information.',
    func: async () => 'Good bye.',
  });

  const tools = [searchTool, retrieverTool, endConversationTool];

  // 2. initialiste the llm and set streaming to true
  const llm = new ChatOpenAI({
    model: 'gpt-4o',
    streaming: true,
  });

  // 3. initialize the agent
  // 3.1 build initial prompt messages (agent_scratchpad is an internal thing that has to be used in order for the agent to correctly work)
  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are a helpful assistant'],
    ['placeholder', '{chat_history}'],
    ['human', '{input}'],
    ['placeholder', '{agent_scratchpad}'],
  ]);

  // 3.2 Build chain including llm, tools that can be called and the corresponding prompt
  const agent = await createOpenAIFunctionsAgent({
    llm,
    tools,
    prompt,
  });

  // 3.3 Runtime for actually using the agent (from v0.2 it is officially deprecated and it is recommended to use langGraph directly)
  const agentExecutor = new AgentExecutor({
    agent,
    tools,
    //verbose: true,
  });

  // 4. Initialise message history and runnable that automatically updates the history with recent messages
  const messageHistory = new ChatMessageHistory([
    new HumanMessage('Can LangSmith help test my LLM applications?'),
    new AIMessage('Yes!'),
  ]);
  const agentWithChatHistory = new RunnableWithMessageHistory({
    runnable: agentExecutor,
    // This is needed because in most real world scenarios, a session id is needed per user.
    // It isn't really used here because we are using a simple in memory ChatMessageHistory.
    // e.g. for using redis:
    // getMessageHistory: RedisChatMessageHistory(session_id, url=REDIS_URL, ttl=600),
    getMessageHistory: _sessionId => messageHistory,
    inputMessagesKey: 'input',
    historyMessagesKey: 'chat_history',
  });

  // 5. Create handlers that are listening for specific events fired by the model
  const toolHandlerCallBack = new ToolCallbackHandler();
  const streamCallbackHandler = new StreamCallbackHandler();

  // 6. First test
  const result = await agentWithChatHistory.invoke(
    {
      input: 'Tell me how',
    },
    // This is needed because in most real world scenarios, a session id is needed per user.
    // It isn't really used here because we are using a simple in memory ChatMessageHistory.
    { configurable: { sessionId: 'foo' }, callbacks: [toolHandlerCallBack, streamCallbackHandler] }
  );
  console.log(result.output);

  // 7. Second test
  const result2 = await agentWithChatHistory.invoke(
    {
      input: 'How is the weather today in Berlin (Germany)?',
    },
    // This is needed because in most real world scenarios, a session id is needed per user.
    // It isn't really used here because we are using a simple in memory ChatMessageHistory.
    { configurable: { sessionId: 'foo' }, callbacks: [toolHandlerCallBack, streamCallbackHandler] }
  );
  console.log(result2.output);

  // 8. Third test
  const result3 = await agentWithChatHistory.invoke(
    {
      input: 'Thank you. Thats it from my side. Good bye!',
    },
    // This is needed because in most real world scenarios, a session id is needed per user.
    // It isn't really used here because we are using a simple in memory ChatMessageHistory.
    { configurable: { sessionId: 'foo' }, callbacks: [toolHandlerCallBack, streamCallbackHandler] }
  );
  console.log(result3.output);

  // 9. Show what tools were called by the agent
  const calledTools = toolHandlerCallBack.getCalledTools();
  console.log(calledTools.keys());
}
