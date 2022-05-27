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

        const query = window.location.search;
        const engine = new Engine(params);
        engine.init().then(() => {
            if (query === '?cloth') engine.renderer.setupCloth();
            else if (query === '?particles') engine.renderer.setupParticles();
            else engine.renderer.setupCubes();
            // engine.renderer.setupParticles();
        });

        // worker.postMessage(data);
        // return () => worker.terminate();
    }, []);

    const base = `${location.origin}/${location.pathname}`;

    return (
        <div>
            <a href={base}>Cubes</a>
            <a href={base + '?cloth'}>Cloth</a>
            <a href={base + '?particles'}>Particles</a>
        </div>
    );
};

ReactDOM.render(<App />, document.getElementById('app'));
