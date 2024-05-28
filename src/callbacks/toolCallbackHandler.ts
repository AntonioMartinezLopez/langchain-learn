import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AgentAction } from 'langchain/agents';

type CalledTools = Map<string, { tool: string; toolInput: string | Record<string, any>; log: string }>;

export class ToolCallbackHandler extends BaseCallbackHandler {
  name = 'tool_callback_handler';
  calledTools: CalledTools;

  constructor() {
    super();
    this.calledTools = new Map();
  }

  handleAgentAction(action: AgentAction) {
    console.log('handleAgentAction', action);
    this.calledTools.set(action.tool, action);
  }

  getCalledTools() {
    return this.calledTools;
  }
}
