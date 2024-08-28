import ort from 'onnxruntime-web/webgpu';
import { loadPyodide } from "pyodide";
import { encode } from "wav-encoder";

const MODELS = {
    audio_separator: [
        {
            name: "UVR_MDXNET_Inst_Main",
            url: "https://huggingface.co/seanghay/uvr_models/resolve/main/UVR-MDX-NET-Inst_Main.onnx",
            size: 52.8,
        }
    ],
};

const config = getConfig();

ort.env.wasm.wasmPaths = 'dist/';
ort.env.wasm.numThreads = config.threads;
// ort.env.wasm.proxy = config.provider == "wasm";

let audioBuffer = null;
let vocalBlob, bgmBlob;

function log(i) {
    const newLine = document.createElement('div');
    newLine.innerText = i;
    newLine.id = i.trim();
    document.getElementById('status').appendChild(newLine);
}

function removeLog(i) {
    const line = document.getElementById(i);
    if (line) {
        line.remove();
    }
}

function updateLatency(i) {
    document.getElementById('inference_latency').innerText = i;
}

/**
 * create config from url
 */
function getConfig() {
    const query = window.location.search.substring(1);
    var config = {
        model: "audio_separator",
        provider: "webgpu",
        device: "gpu",
        threads: "1",
    };
    let vars = query.split("&");
    for (var i = 0; i < vars.length; i++) {
        let pair = vars[i].split("=");
        if (pair[0] in config) {
            config[pair[0]] = decodeURIComponent(pair[1]);
        } else if (pair[0].length > 0) {
            throw new Error("unknown argument: " + pair[0]);
        }
    }
    config.threads = parseInt(config.threads);
    config.local = parseInt(config.local);
    return config;
}

/*
 * fetch and cache url
 */
async function fetchAndCache(url, name) {
    try {
        const cache = await caches.open("onnx");
        let cachedResponse = await cache.match(url);
        if (cachedResponse == undefined) {
            await cache.add(url);
            cachedResponse = await cache.match(url);
            log(`${name} (Network)\n\n`);
        } else {
            log(`${name} (Cached)\n\n`);
        }
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`${name} (Network)\n\n`);
        return await fetch(url).then(response => response.arrayBuffer());
    }
}

/*
 * load models one at a time
 */
async function load_models(models) {
    const cache = await caches.open("onnx");
    let missing = 0;
    for (const [name, model] of Object.entries(models)) {
        let cachedResponse = await cache.match(model.url);
        if (cachedResponse === undefined) {
            missing += model.size;
        }
    }
    if (missing > 0) {
        log(`Downloading ${missing} MB from network... It might take a while\n\n`);
    } else {
        log("Loading...\n\n");
    }
    const start = performance.now();
    for (const [name, model] of Object.entries(models)) {
        try {
            const opt = {
                executionProviders: [config.provider],
                enableMemPattern: false,
                enableCpuMemArena: false,
                extra: {
                    session: {
                        disable_prepacking: "1",
                        use_device_allocator_for_initializers: "1",
                        use_ort_model_bytes_directly: "1",
                        use_ort_model_bytes_for_initializers: "1"
                    }
                },
            };
            const model_bytes = await fetchAndCache(model.url, model.name);
            const extra_opt = model.opt || {};
            const sess_opt = { ...opt, ...extra_opt };
            model.sess = await ort.InferenceSession.create(model_bytes, sess_opt);
        } catch (e) {
            log(`${model.url} failed, ${e}\n\n`);
        }
    }
    const stop = performance.now();
    removeLog(`Downloading ${missing} MB from network... It might take a while`);
    removeLog("Loading...");
    log(`Model Ready: ${(stop - start).toFixed(2)} ms\n\n`);
}

/**
 * Handle input audio and decode
 */
async function handleAudio(file) {
    const reader = new FileReader();
    reader.onload = async function(event) {
        const arrayBuffer = event.target.result;
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
        audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        removeLog("Audio file uploaded");
        log("Audio file uploaded\n\n");
    };
    reader.readAsArrayBuffer(file);
}

async function processAudio() {
    if (!audioBuffer) {
        log("Please upload an audio file first\n\n");
        return;
    } else {
        removeLog("Please upload an audio file first");
        removeLog("Audio file uploaded");
        log("Processing audio...\n\n");
        updateLatency(``);
    }
    const session = await MODELS[config.model][0].sess;

    const numberOfChannels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;
    const sampleRate = audioBuffer.sampleRate;

    // Extract audio data from audioBuffer
    let audioData = [];
    for (let channel = 0; channel < numberOfChannels; channel++) {
        audioData.push(audioBuffer.getChannelData(channel));
    }

    try {
        let pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/"
        });
    
        await pyodide.loadPackage(['numpy', 'scipy']);
    
        const response = await fetch('separate.py');
        const pythonCode = await response.text();
        await pyodide.runPythonAsync(pythonCode);
    
        const Separator = pyodide.globals.get('Separator');
        const np = pyodide.pyimport('numpy');

        if (Separator) {
            const args = pyodide.toPy({
                denoise: true,
                margin: 44100,
                chunks: 15,
                n_fft: 6144,
                dim_t: 8,
                dim_f: 2048
            });

            const separatorInstance = Separator(args);
            const audioDataNdarray = np.array(audioData);

            const segmentedMix = separatorInstance.segment(audioDataNdarray);
            const segmentedMix_js = segmentedMix.toJs();

            const start = performance.now();

            let opt = [[], []];
            for (let [skip, cmix] of segmentedMix_js) {
                const cmixNdarray = np.array(cmix)
                const modelInput = separatorInstance.preprocess(cmixNdarray);
                let buffer = modelInput.getBuffer("f32");
                modelInput.destroy();
                const inputTensor = new ort.Tensor(new Float32Array(buffer.data), buffer.shape);
                let specPred;
                if (args.toJs().denoise) {
                    const negInput = new ort.Tensor(inputTensor.data.map(x => -x), inputTensor.dims);
                    const negResult = await session.run({ "input": negInput });
                    const posResult = await session.run({ "input": inputTensor });
            
                    specPred = negResult.output.data.map((val, idx) => -val * 0.5 + posResult.output.data[idx] * 0.5);
                } else {
                    const result = await session.run({ "input": inputTensor });
                    specPred = result.output.data;
                }
                const specPredNdarray = np.array(specPred);
                const sources = separatorInstance.postprocess(skip, specPredNdarray);
                const sources_js = sources.toJs();

                opt[0] = opt[0].concat(Array.from(sources_js[0]));
                opt[1] = opt[1].concat(Array.from(sources_js[1]));

                buffer.release();
            }
            segmentedMix.destroy();

            let res = audioData.map((mixItem, index) => {
                return mixItem.map((item, idx) => {
                    return item - opt[index][idx];
                });
            });

            const stop = performance.now();
            removeLog("Processing audio...");
            updateLatency(`${(stop - start).toFixed(2)} ms`);

            let audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            let vocalBuffer = audioCtx.createBuffer(2, res[0].length, audioCtx.sampleRate);
            let bgmBuffer = audioCtx.createBuffer(2, opt[0].length, audioCtx.sampleRate);
            
            for (let channel = 0; channel < 2; channel++) {
                let vocalData = vocalBuffer.getChannelData(channel);
                let bgmData = bgmBuffer.getChannelData(channel);
                for (let i = 0; i < vocalData.length; i++) {
                    vocalData[i] = res[channel][i];
                    bgmData[i] = opt[channel][i];
                }
            }

            let vocalArrayBuffer = await encode({
                sampleRate: audioCtx.sampleRate,
                channelData: [vocalBuffer.getChannelData(0), vocalBuffer.getChannelData(1)]
            });
            let bgmArrayBuffer = await encode({
                sampleRate: audioCtx.sampleRate,
                channelData: [bgmBuffer.getChannelData(0), bgmBuffer.getChannelData(1)]
            });
            
            vocalBlob = new Blob([vocalArrayBuffer], { type: "audio/wav" });
            bgmBlob = new Blob([bgmArrayBuffer], { type: "audio/wav" });
            
            let vocalUrl = URL.createObjectURL(vocalBlob);
            let bgmUrl = URL.createObjectURL(bgmBlob);
            
            document.getElementById("download-vocal-button").href = vocalUrl;
            document.getElementById("download-bgm-button").href = bgmUrl;

            document.getElementById("download-vocal-button").download = "vocal.wav";
            document.getElementById("download-bgm-button").download = "bgm.wav";
        } else {
            log("Failed to load WASM Class\n\n");
        }
    } catch (error) {
        log(`${error}\n\n`);
    }
}

function downloadAudio(audioBlob, filename) {
    let url = URL.createObjectURL(audioBlob);
    let a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
}

async function main() {
    const filein = document.getElementById("file-in");

    await load_models(MODELS[config.model]).then(() => {
        document.getElementById("process-button").addEventListener("click", processAudio);
        document.getElementById("download-vocal-button").addEventListener('click', function() {
            downloadAudio(vocalBlob, 'vocal.wav');
        });
        document.getElementById("download-bgm-button").addEventListener('click', function() {
            downloadAudio(bgmBlob, 'bgm.wav');
        });

        filein.onchange = function (evt) {
            let target = evt.target || window.event.srcElement;
            let files = target.files;
            if (FileReader && files && files.length) {
                handleAudio(files[0]);
            }
        };
    }, (e) => {
        log(e);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    main();
});
