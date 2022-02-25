import { useEffect } from 'react';
import ReactDOM from 'react-dom';
import Engine from './engine/core/engine';
import { EngineParam } from './engine/types';

import './engine/ui/style/index.css';

const App = () => {
    useEffect(() => {
        // const worker = new Worker(new URL('./engine/worker.ts', import.meta.url));

        // worker.addEventListener('message', e => {
        //     console.log(e.data);
        // });

        const canvas = document.getElementById('webgpu') as HTMLCanvasElement;
        // const offscreen = canvas.transferControlToOffscreen();

        const params: EngineParam = {
            canvas,
            screen: {
                width: screen.width,
                height: screen.height,
            },
            dpr: window.devicePixelRatio,
        };

        const engine = new Engine(params);
        engine.init();

        // worker.postMessage(data);

        // return () => worker.terminate();
    }, []);

    return <></>;
};

ReactDOM.render(<App />, document.getElementById('app'));
