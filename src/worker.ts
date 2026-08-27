import { pipeline, env, AutomaticSpeechRecognitionPipeline } from '@xenova/transformers';

// Skip local check for models (it will try to fetch from HuggingFace)
env.allowLocalModels = false;

class TranscriptionPipeline {
    static task = 'automatic-speech-recognition' as const;
    static model = 'Xenova/whisper-base';
    static instance: Promise<AutomaticSpeechRecognitionPipeline> | null = null;

    static async getInstance(progress_callback?: (progress: any) => void) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, { progress_callback }) as Promise<AutomaticSpeechRecognitionPipeline>;
        }
        return this.instance;
    }
}

self.onmessage = async (event: MessageEvent) => {
    const { audio, language, task } = event.data;

    try {
        console.log(`Starting ${task} for language: ${language}`);
        const transcriber = await TranscriptionPipeline.getInstance((x) => {
            // Only send serializable progress data
            self.postMessage({
                status: x.status,
                file: x.file,
                progress: x.progress,
                loaded: x.loaded,
                total: x.total
            });
        });

        const output = await transcriber(audio, {
            language: language,
            task: task, // 'transcribe' or 'translate'
            chunk_length_s: 30,
            stride_length_s: 5,
            callback_function: (x: any) => {
                // Only send text updates
                self.postMessage({
                    status: 'update',
                    text: x.text
                });
            }
        } as any);

        const result = output as any;
        self.postMessage({
            status: 'complete',
            output: {
                text: result?.text || '',
                chunks: result?.chunks ? result.chunks.map((c: any) => ({
                    text: c.text,
                    timestamp: c.timestamp
                })) : []
            }
        });
    } catch (error: any) {
        self.postMessage({
            status: 'error',
            message: error.message
        });
    }
};
