// Push-to-talk microphone capture -> 16 kHz mono Float32 for Whistle.

const MAX_SECONDS = 15;

export class Recorder {
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    this.ctx = new AudioContext();
    const src = this.ctx.createMediaStreamSource(this.stream);
    // ScriptProcessor is deprecated but universally available and fine for a few seconds of audio
    this.node = this.ctx.createScriptProcessor(4096, 1, 1);
    this.chunks = [];
    this.node.onaudioprocess = (e) => {
      this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      if (this.chunks.length * 4096 > MAX_SECONDS * this.ctx.sampleRate) this.onLimit?.();
    };
    src.connect(this.node);
    this.node.connect(this.ctx.destination);
  }

  /** Stop and return 16 kHz mono samples in [-1, 1]. */
  async stop() {
    this.node.disconnect();
    this.stream.getTracks().forEach((t) => t.stop());
    const rate = this.ctx.sampleRate;
    await this.ctx.close();
    const n = this.chunks.reduce((a, c) => a + c.length, 0);
    if (n < rate * 0.25) return null;                  // under a quarter second: a mis-click
    const buf = new AudioBuffer({ length: n, numberOfChannels: 1, sampleRate: rate });
    const data = buf.getChannelData(0);
    let off = 0;
    for (const c of this.chunks) { data.set(c, off); off += c.length; }
    const off16 = new OfflineAudioContext(1, Math.ceil(n * 16000 / rate), 16000);
    const s = off16.createBufferSource();
    s.buffer = buf;
    s.connect(off16.destination);
    s.start();
    const out = await off16.startRendering();
    return out.getChannelData(0).slice();
  }
}
