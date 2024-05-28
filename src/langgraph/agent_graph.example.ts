import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { ToolExecutor } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { DynamicTool } from 'langchain/tools';
import { createRetrieverTool } from 'langchain/tools/retriever';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { convertToOpenAIFunction } from '@langchain/core/utils/function_calling';
import { BaseMessage, FunctionMessage, HumanMessage } from '@langchain/core/messages';
import { AgentAction } from 'langchain/agents';
import { END, START, StateGraph, StateGraphArgs } from '@langchain/langgraph';

export async function agentGraphExample() {
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

  // 1.4 Add tools to the ToolExecutor: This is a real simple class that takes in a ToolInvocation and calls that tool, returning the output.
  const tools = [searchTool, retrieverTool, endConversationTool];
  const toolExecutor = new ToolExecutor({ tools });

  const model = new ChatOpenAI({
    model: 'gpt-4o',
    streaming: true,
  });

  const toolsAsOpenAIFunctions = tools.map(tool => convertToOpenAIFunction(tool));
  const newModel = model.bind({
    functions: toolsAsOpenAIFunctions,
  });

  interface IState {
    messages: BaseMessage[];
  }

  // This defines the agent state
  const graphState: StateGraphArgs<IState>['channels'] = {
    messages: {
      value: (x: BaseMessage[], y: BaseMessage[]) => x.concat(y),
      default: () => [],
    },
  };

  // Define the function that determines whether to continue or not
  const shouldContinue = (state: { messages: Array<BaseMessage> }) => {
    const { messages } = state;
    const lastMessage = messages[messages.length - 1];
    // If there is no function call, then we finish
    if (!lastMessage?.additional_kwargs?.function_call || !lastMessage?.additional_kwargs.function_call) {
      return 'end';
    }
    // Otherwise if there is, we continue
    return 'continue';
  };

  // Define the function to execute tools
  const _getAction = (state: { messages: Array<BaseMessage> }): AgentAction => {
    const { messages } = state;
    // Based on the continue condition
    // we know the last message involves a function call
    const lastMessage = messages[messages.length - 1];
    if (!lastMessage) {
      throw new Error('No messages found.');
    }
    if (!lastMessage.additional_kwargs.function_call) {
      throw new Error('No function call found in message.');
    }
    // We construct an AgentAction from the function_call
    return {
      tool: lastMessage.additional_kwargs.function_call.name,
      toolInput: JSON.stringify(lastMessage.additional_kwargs.function_call.arguments),
      log: '',
    };
  };

  // Define the function that calls the model
  const callModel = async (state: { messages: Array<BaseMessage> }) => {
    const { messages } = state;
    // console.log('State messages: ', messages);
    const response = await newModel.invoke(messages);
    // console.log('response: ', response);
    // We return a list, because this will get added to the existing list
    return {
      messages: [response],
    };
  };

  const callTool = async (state: { messages: Array<BaseMessage> }) => {
    const action = _getAction(state);
    // We call the tool_executor and get back a response
    const response = await toolExecutor.invoke(action);
    // We use the response to create a FunctionMessage
    const functionMessage = new FunctionMessage({
      content: response,
      name: action.tool,
    });
    // We return a list, because this will get added to the existing list
    return { messages: [functionMessage] };
  };

  // Define a new graph
  const workflow = new StateGraph({
    channels: graphState,
  })
    .addNode('agent', callModel)
    .addNode('action', callTool);

  // Set the entrypoint as `agent`
  // This means that this node is the first one called
  workflow.addEdge(START, 'agent');

  // We now add a conditional edge
  workflow.addConditionalEdges(
    // First, we define the start node. We use `agent`.
    // This means these are the edges taken after the `agent` node is called.
    'agent',
    // Next, we pass in the function that will determine which node is called next.
    shouldContinue,
    // Finally we pass in a mapping.
    // The keys are strings, and the values are other nodes.
    // END is a special node marking that the graph should finish.
    // What will happen is we will call `should_continue`, and then the output of that
    // will be matched against the keys in this mapping.
    // Based on which one it matches, that node will then be called.
    {
      // If `tools`, then we call the tool node.
      continue: 'action',
      // Otherwise we finish.
      end: END,
    }
  );

  // We now add a normal edge from `tools` to `agent`.
  // This means that after `tools` is called, `agent` node is called next.
  workflow.addEdge('action', 'agent');

  // Finally, we compile it!
  // This compiles it into a LangChain Runnable,
  // meaning you can use it as you would any other runnable
  const app = workflow.compile();

  const inputs = {
    messages: [new HumanMessage('what is the weather in Berlin (Germany)?')],
  };

  const result = await app.invoke(inputs);
  console.log('RESULT:', result);

  const inputs2 = {
    messages: [...result.messages, new HumanMessage('Can you tell me a bit about the history of the city?')],
  };

  console.log('Graphstate: ', graphState.messages);
  const result2 = await app.invoke(inputs2);
  console.log(result2);
}
