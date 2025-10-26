import AsteroidASM from './wasm';

const ASM = new AsteroidASM();
const ctx: Worker = self as any; // eslint-disable-line no-restricted-globals

const gcWorker = new Worker(new URL('./gc.ts', import.meta.url));

const gc = (...args: Transferable[]) => {
    gcWorker.postMessage({ args }, args);
};

const Buffers: Record<string, ArrayBuffer> = {
    posX: null,
    posY: null,
    state: null,
};

const copyBuffer = (name: string, ptr: number, size: number) => {
    if (!Buffers[name] || Buffers[name].byteLength < size) {
        Buffers[name] && gc(Buffers[name]);
        Buffers[name] = new ArrayBuffer(size);
    }
    const source = new Uint8Array(ASM.memory.buffer, ptr, size);
    new Uint8Array(Buffers[name]).set(source);
};

const tick = (sim = true, vel = -1 / 30) => {
    const size = ASM.module.get_asteroid_size();
    const posXPtr = ASM.module.get_asteroid_pos_x();
    const posYPtr = ASM.module.get_asteroid_pos_y();
    const statePtr = ASM.module.get_asteroid_state();

    copyBuffer('posX', posXPtr, size * Int32Array.BYTES_PER_ELEMENT);
    copyBuffer('posY', posYPtr, size * Int32Array.BYTES_PER_ELEMENT);
    copyBuffer('state', statePtr, size * Uint32Array.BYTES_PER_ELEMENT);

    ctx.postMessage({ event: 'tick', buffers: Buffers, size }, Object.values(Buffers));
    Buffers.posX = null;
    Buffers.posY = null;
    Buffers.state = null;

    sim && ASM.module.tick(vel);
};

ctx.addEventListener('message', ({ data }) => {
    if (!ASM.ready) return console.error('Worker not ready', data);

    const { call, args, buffers } = data;
    if (call === 'start') tick(false);
    else if (call === 'sim') {
        if (Buffers.posX || Buffers.posY || Buffers.state)
            return console.error('Buffers not cleared???');
        Buffers.posX = buffers.posX;
        Buffers.posY = buffers.posY;
        Buffers.state = buffers.state;
        tick(true, args[0]);
    } else console.error('Unknown call', call, args);
});

(async () => {
    await ASM.init();
    ctx.postMessage({ event: 'ready' });
})();
