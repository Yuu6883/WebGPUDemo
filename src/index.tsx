import { useEffect } from 'react';
import ReactDOM from 'react-dom';
import { EngineParam } from './engine/types';

const App = () => {
    useEffect(() => {
        const worker = new Worker(new URL('./engine/worker.ts', import.meta.url));

        worker.addEventListener('message', e => {
            console.log(e.data);
        });

        const canvas = document.getElementById('canvas') as HTMLCanvasElement;
        const offscreen = canvas.transferControlToOffscreen();

        const data: EngineParam = {
            canvas: offscreen,
            screen: {
                width: screen.width,
                height: screen.height,
            },
            dpr: window.devicePixelRatio,
        };

        worker.postMessage(data, [offscreen]);

        return () => worker.terminate();
    }, []);

    return <></>;
};

ReactDOM.render(<App />, document.getElementById('app'));
