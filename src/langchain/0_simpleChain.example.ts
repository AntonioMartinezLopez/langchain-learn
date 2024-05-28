import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';

export async function simpleChainExample() {
  // 1. define prompt
  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are a world class technical documentation writer.'],
    ['user', '{input}'],
  ]);

  // 2. create model
  const chatModel = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4o',
  });

  // 3. define output parser
  const outputParser = new StringOutputParser();

  // 4. chain everything together
  const llmChain = prompt.pipe(chatModel).pipe(outputParser);

  console.log(await llmChain.invoke({ input: 'what is LangChain?' }));
}
