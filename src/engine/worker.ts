import Engine from './core/engine';
import { EngineParam } from './types';

const ctx: Worker = self as any;

ctx.addEventListener('message', (e: MessageEvent<EngineParam>) => {
    const { data } = e;
    const engine = new Engine(data);
    engine.init();
});
