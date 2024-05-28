import { BaseCallbackHandler } from '@langchain/core/callbacks/base';

export class StreamCallbackHandler extends BaseCallbackHandler {
  name = 'stream_callback_handler';

  handleLLMNewToken(token: string) {
    console.log({ token });
  }

  constructor() {
    super();
  }
}
