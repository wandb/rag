import { StreamProcessorSrc } from './worklets/stream_processor.js';
import { AudioAnalysis } from './analysis/audio_analysis.js';

/**
 * Plays audio streams received in raw PCM16 chunks from the browser
 * @class
 */
export class WavStreamPlayer {
  /**
   * Creates a new WavStreamPlayer instance
   * @param {{sampleRate?: number}} options
   * @returns {WavStreamPlayer}
   */
  constructor({ sampleRate = 44100 } = {}) {
    this.scriptSrc = StreamProcessorSrc;
    this.sampleRate = sampleRate;
    this.context = null;
    this.stream = null;
    this.analyser = null;
    this.trackSampleOffsets = {};
    this.interruptedTrackIds = {};
    this.mediaRecorder = null;
    this.audioElement = null;
  }

  /**
   * Connects the audio context and sets up MediaRecorder
   * @param {HTMLAudioElement} audioElement
   * @returns {Promise<true>}
   */
  async connect(audioElement) {
    this.audioElement = audioElement;
    this.context = new AudioContext({ sampleRate: this.sampleRate });
    if (this.context.state === 'suspended') {
      await this.context.resume();
    }

    // Create and configure analyzer
    const analyser = this.context.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.8;
    analyser.minDecibels = -90;
    analyser.maxDecibels = -10;
    this.analyser = analyser;

    // Create initial MediaElementSource connection
    this.sourceNode = this.context.createMediaElementSource(audioElement);
    this.sourceNode.connect(this.analyser);
    this.sourceNode.connect(this.context.destination);

    // Create a MediaStream from the audio context
    const dest = this.context.createMediaStreamDestination();

    try {
      await this.context.audioWorklet.addModule(this.scriptSrc);
    } catch (e) {
      console.error(e);
      throw new Error(`Could not add audioWorklet module: ${this.scriptSrc}`);
    }

    // Set up MediaRecorder
    this.mediaRecorder = new MediaRecorder(dest.stream);
    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        const url = URL.createObjectURL(event.data);
        // Clean up the old URL
        const oldUrl = this.audioElement.dataset.blobUrl;
        if (oldUrl) {
          URL.revokeObjectURL(oldUrl);
        }
        this.audioElement.src = url;
        this.audioElement.dataset.blobUrl = url;
      }
    };

    // Start recording
    this.mediaRecorder.start(100);
    return true;
  }

  /**
   * Gets the current frequency domain data from the playing track
   * @param {"frequency"|"music"|"voice"} [analysisType]
   * @param {number} [minDecibels] default -100
   * @param {number} [maxDecibels] default -30
   * @returns {import('./analysis/audio_analysis.js').AudioAnalysisOutputType}
   */
  getFrequencies(
    analysisType = 'frequency',
    minDecibels = -100,
    maxDecibels = -30
  ) {
    if (!this.analyser) {
      throw new Error('Not connected, please call .connect() first');
    }
    return AudioAnalysis.getFrequencies(
      this.analyser,
      this.sampleRate,
      null,
      analysisType,
      minDecibels,
      maxDecibels
    );
  }

  /**
   * Starts audio streaming
   * @private
   * @returns {Promise<true>}
   */
  _start() {
    const streamNode = new AudioWorkletNode(this.context, 'stream_processor');
    streamNode.connect(this.context.destination);
    streamNode.connect(this.analyser);

    // Also connect to the MediaStreamDestination
    const dest = this.context.createMediaStreamDestination();
    streamNode.connect(dest);

    streamNode.port.onmessage = (e) => {
      const { event } = e.data;
      if (event === 'stop') {
        streamNode.disconnect();
        this.stream = null;
      } else if (event === 'offset') {
        const { requestId, trackId, offset } = e.data;
        const currentTime = offset / this.sampleRate;
        this.trackSampleOffsets[requestId] = { trackId, offset, currentTime };
      }
    };
    this.stream = streamNode;
    return true;
  }

  /**
   * Adds 16BitPCM data to the currently playing audio stream
   * You can add chunks beyond the current play point and they will be queued for play
   * @param {ArrayBuffer|Int16Array} arrayBuffer
   * @param {string} [trackId]
   * @returns {Int16Array}
   */
  add16BitPCM(arrayBuffer, trackId = 'default') {
    if (typeof trackId !== 'string') {
      throw new Error(`trackId must be a string`);
    } else if (this.interruptedTrackIds[trackId]) {
      return;
    }
    if (!this.stream) {
      this._start();
    }
    let buffer;
    if (arrayBuffer instanceof Int16Array) {
      buffer = arrayBuffer;
    } else if (arrayBuffer instanceof ArrayBuffer) {
      buffer = new Int16Array(arrayBuffer);
    } else {
      throw new Error(`argument must be Int16Array or ArrayBuffer`);
    }
    this.stream.port.postMessage({ event: 'write', buffer, trackId });
    return buffer;
  }

  /**
   * Gets the offset (sample count) of the currently playing stream
   * @param {boolean} [interrupt]
   * @returns {{trackId: string|null, offset: number, currentTime: number}}
   */
  async getTrackSampleOffset(interrupt = false) {
    if (!this.stream) {
      return null;
    }
    const requestId = crypto.randomUUID();
    this.stream.port.postMessage({
      event: interrupt ? 'interrupt' : 'offset',
      requestId,
    });
    let trackSampleOffset;
    while (!trackSampleOffset) {
      trackSampleOffset = this.trackSampleOffsets[requestId];
      await new Promise((r) => setTimeout(() => r(), 1));
    }
    const { trackId } = trackSampleOffset;
    if (interrupt && trackId) {
      this.interruptedTrackIds[trackId] = true;
    }
    return trackSampleOffset;
  }

  /**
   * Strips the current stream and returns the sample offset of the audio
   * @param {boolean} [interrupt]
   * @returns {{trackId: string|null, offset: number, currentTime: number}}
   */
  async interrupt() {
    return this.getTrackSampleOffset(true);
  }

  /**
   * Resets the player and starts a new MediaRecorder instance
   * @returns {Promise<void>}
   */
  async reset() {
    // Stop and clean up existing stream
    if (this.stream) {
      this.stream.port.postMessage({ event: 'reset' });
      this.stream.disconnect();
      this.stream = null;
    }

    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }

    // Reset all tracking variables
    this.trackSampleOffsets = {};
    this.interruptedTrackIds = {};

    // Clean up old blob URL and reset audio element
    if (this.audioElement) {
      const oldUrl = this.audioElement.dataset.blobUrl;
      if (oldUrl) {
        URL.revokeObjectURL(oldUrl);
        delete this.audioElement.dataset.blobUrl;
      }

      // Fully reset the audio element
      this.audioElement.pause();
      this.audioElement.currentTime = 0;
      this.audioElement.src = '';
      this.audioElement.load();  // Force browser to clear buffer
    }

    // Create new MediaRecorder with fresh stream
    const dest = this.context.createMediaStreamDestination();
    this.mediaRecorder = new MediaRecorder(dest.stream);
    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        const url = URL.createObjectURL(event.data);
        if (this.audioElement) {
          // Clean up old URL before setting new one
          const oldUrl = this.audioElement.dataset.blobUrl;
          if (oldUrl) {
            URL.revokeObjectURL(oldUrl);
          }
          this.audioElement.src = url;
          this.audioElement.dataset.blobUrl = url;
        }
      }
    };

    // Start recording with smaller time slice for more responsive updates
    this.mediaRecorder.start(100);
  }
}

globalThis.WavStreamPlayer = WavStreamPlayer;
