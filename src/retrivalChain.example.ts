import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';

export async function retrivalChainExample() {
  // 1. Build Retrival chain element
  // 1.1 Fetch document
  const loader = new CheerioWebBaseLoader('https://docs.smith.langchain.com/user_guide');
  const docs = await loader.load();
  console.log(docs.length);
  console.log(docs[0]?.pageContent.length);
  // 1.2 Initialise splitter
  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments(docs);
  console.log(splitDocs.length);
  console.log(splitDocs[0]?.pageContent.length);
  // 1.3 create vectorstore and prepare to be used for retrival
  const embeddings = new OpenAIEmbeddings();
  const vectorstore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
  const retriever = vectorstore.asRetriever();

  // 2. Create a chain that accepts an array of documents and passes them to a model
  // 2.1 Define Prompt template
  const prompt = ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:
  
    <context>
    {context}
    </context>
  
    Question: {input}`);

  // 2.2 Create the chain and add prompt
  const chatModel = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4o',
  });
  const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
  });

  // 4. Connect retriever and document accepting model chain to one chain
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });

  console.log((await retrievalChain.invoke({ input: 'what is LangChain?' })).answer);
}
