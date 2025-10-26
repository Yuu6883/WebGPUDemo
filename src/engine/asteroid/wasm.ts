import { AsteroidWebAssemblyModule } from '../types';

const textDecoder = new TextDecoder('utf-8');

export default class AsteroidASM {
    public ready = false;
    readonly memory = new WebAssembly.Memory({ initial: 16384, maximum: 16384 });

    private stdout_buffer = '';
    private stderr_buffer = '';
    public module: AsteroidWebAssemblyModule;

    flush(fd: number) {
        if (fd === 1) {
            let i = 0;
            while (i < this.stdout_buffer.length) {
                const next_newline = this.stdout_buffer.indexOf('\n', i);
                if (next_newline === -1) {
                    this.stdout_buffer = this.stdout_buffer.slice(i);
                    break;
                }
                console.log(this.stdout_buffer.slice(i, next_newline));
                i = next_newline + 1;
            }
            if (i === this.stdout_buffer.length) {
                this.stdout_buffer = '';
            }
        } else if (fd === 2) {
            let i = 0;
            while (i < this.stderr_buffer.length) {
                const next_newline = this.stderr_buffer.indexOf('\n', i);
                if (next_newline === -1) {
                    this.stderr_buffer = this.stderr_buffer.slice(i);
                    break;
                }
                console.error(this.stderr_buffer.slice(i, next_newline));
                i = next_newline + 1;
            }
            if (i === this.stderr_buffer.length) {
                this.stderr_buffer = '';
            }
        }
    }

    async init() {
        const memory = this.memory;
        // Provide minimal imports. Add more if your wasm expects them.
        const imports = {
            wasi_snapshot_preview1: {
                args_sizes_get: () => 0,
                args_get: () => 0,
                environ_sizes_get: (count: number, buf_size: number) => {
                    const view = new DataView(memory.buffer);
                    view.setUint32(count, 0, true);
                    view.setUint32(buf_size, 0, true);
                    return 0;
                },
                environ_get: () => 0,
                fd_write: (
                    fd: number,
                    iovs_ptr: number,
                    iovs_len: number,
                    result: number,
                ) => {
                    const view = new DataView(memory.buffer);
                    let written = 0;

                    for (let i = 0; i < iovs_len; i++) {
                        const bufPtr = view.getUint32(iovs_ptr + i * 8, true);
                        const bufLen = view.getUint32(iovs_ptr + i * 8 + 4, true);

                        const bytes = new Uint8Array(memory.buffer, bufPtr, bufLen);
                        const str = textDecoder.decode(bytes);

                        if (fd === 1) {
                            this.stdout_buffer += str;
                            this.flush(1);
                        } else if (fd === 2) {
                            this.stderr_buffer += str;
                            this.flush(2);
                        } else {
                            return 8; // __WASI_ERRNO_BADF
                        }

                        written += bufLen;
                    }
                    view.setUint32(result, written, true);
                    return 0;
                },
                fd_seek: () => {
                    console.log(`fd_seek called: ${arguments}`);
                },
                fd_read: () => {
                    console.log(`fd_read called: ${arguments}`);
                },
                fd_close: () => {
                    console.log(`fd_close called: ${arguments}`);
                },
                proc_exit: () => 0,
                clock_time_get: (_, __, out: number) => {
                    out = out >>> 0;
                    const ts = BigInt(new Date().getTime()) * 1000000n;
                    const view = new DataView(memory.buffer);
                    view.setBigUint64(out, ts, true);
                    return 0;
                },
            },
            env: {
                memory,
                __main_argc_argv: () => 0,
                emscripten_notify_memory_growth: (...args) => {
                    console.log(
                        (memory.buffer.byteLength / 1024 / 1024).toFixed(3) + 'MB',
                    );
                    console.log(
                        `emscripten_notify_memory_growth called: ${JSON.stringify(args)}`,
                    );
                },
            },
        };

        try {
            const wasmPath = 'assets/asteroid.wasm';
            const { instance } = await WebAssembly.instantiateStreaming(
                fetch(wasmPath),
                imports,
            );
            // Call run_bench and print the result (if any)

            const mod = (this.module = instance.exports as AsteroidWebAssemblyModule);
            try {
                mod._start();
            } catch (e) {}
            // mod.run_bench();

            mod.init_map();
            mod.set_asteroid_size(512 * 1024);
            console.log('set_asteroid_size', mod.get_asteroid_size());
            mod.populate_asteroids();

            this.ready = true;
        } catch (err) {
            console.log(
                `memory size: ${(memory.buffer.byteLength / 1024 / 1024).toFixed(3)}MB`,
            );
            console.error('Failed to load or run asteroid.wasm:', err);
        }
    }
}
