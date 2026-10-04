// Planetarium renderer: stereographic projection of the sky in horizontal
// (alt-az) coordinates on a 2D canvas.  All astronomy goes through
// astronomy-engine (global `Astronomy`); star positions are J2000 vectors
// rotated to the observer's horizon once per frame with one matrix.

const A = globalThis.Astronomy;
const DEG = Math.PI / 180;

export const BODIES = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"];
const BODY_STYLE = {
  Sun: ["#ffe9a8", 9], Moon: ["#f2f0e6", 8], Mercury: ["#cdbfb0", 3], Venus: ["#fff6d8", 4.5],
  Mars: ["#ff8a5c", 3.5], Jupiter: ["#f3d9b1", 4.5], Saturn: ["#e9d39a", 4], Uranus: ["#b8f0f0", 2.5],
  Neptune: ["#8fb0ff", 2.5], Pluto: ["#c9b8a8", 1.5],
};

export const OVERLAYS = ["constellation lines", "constellation names", "star names", "planet names",
  "deep sky objects", "grid", "ecliptic", "horizon", "atmosphere"];

const unit = (raDeg, decDeg) => {
  const ra = raDeg * DEG, dec = decDeg * DEG;
  return [Math.cos(dec) * Math.cos(ra), Math.cos(dec) * Math.sin(ra), Math.sin(dec)];
};

// B-V colour index -> rough star colour
function bvColor(bv) {
  const t = Math.max(-0.4, Math.min(2.0, bv));
  if (t < 0) return [170 + t * 100, 191 + t * 60, 255];
  if (t < 0.6) return [230 + t * 40, 235 - t * 20, 255 - t * 90];
  if (t < 1.4) return [255, 240 - (t - 0.6) * 90, 200 - (t - 0.6) * 150];
  return [255, 170 - (t - 1.4) * 60, 80];
}

export class Sky {
  constructor(canvas, catalog) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.cat = catalog;
    this.stars = catalog.stars.map(([ra, dec, mag, bv, name, con]) => ({
      v: unit(ra, dec), ra, dec, mag, name, con, rgb: bvColor(bv).map(Math.round).join(","),
    }));
    this.cons = Object.entries(catalog.cons).map(([abbr, c]) => ({
      abbr, name: c.name, v: unit(c.ra, c.dec), ra: c.ra, dec: c.dec,
      lines: c.lines.map((seg) => seg.map(([ra, dec]) => unit(ra, dec))),
    }));
    this.dso = catalog.dso.map((d) => ({ ...d, v: unit(d.ra, d.dec) }));

    this.place = { name: "San Francisco, California", lat: 37.77, lon: -122.42, tz: "America/Los_Angeles" };
    this.time = new Date();
    this.rate = 1;                     // sky seconds per real second
    this.view = { az: 180, alt: 35, fov: 100 };
    this.anim = null;                  // in-flight slew
    this.followTarget = null;
    this.highlight = null;             // {target, until}
    this.paths = [];                   // [{name, pts: [vec], ticks: [{v,label}]}]
    this.overlays = new Set(["constellation lines", "planet names", "horizon", "atmosphere", "constellation names"]);
    this.onChange = null;              // called with a short description after state changes

    this._installInput();
    this._last = performance.now();
    requestAnimationFrame((t) => this._frame(t));
  }

  // ---- state setters used by the tools -------------------------------------

  observer() { return new A.Observer(this.place.lat, this.place.lon, 0); }

  setPlace(place) { this.place = place; this._recompute(); }
  setTime(date) { this.time = date; this._recompute(); }
  setRate(r) { this.rate = r; }
  setOverlay(name, on) { on ? this.overlays.add(name) : this.overlays.delete(name); }

  slewTo(az, alt, fov = this.view.fov, ms = 1400) {
    const from = { ...this.view };
    let daz = ((az - from.az + 540) % 360) - 180;          // shortest way round
    this.anim = { from, to: { az: from.az + daz, alt, fov }, t0: performance.now(), ms };
  }

  zoomBy(f) { this.slewTo(this.view.az, this.view.alt, Math.max(0.5, Math.min(200, this.view.fov * f)), 600); }

  /** J2000 unit vector -> {az, alt} in degrees for the current time and place. */
  horizontal(v, rot = this._rot()) {
    const h = A.RotateVector(rot, new A.Vector(v[0], v[1], v[2], this._astroTime()));
    const x = h.x, y = h.y, z = h.z;                       // x north, y west, z zenith
    return { az: ((Math.atan2(-y, x) / DEG) + 360) % 360, alt: Math.asin(Math.max(-1, Math.min(1, z))) / DEG };
  }

  bodyVector(name, date = this.time) {
    const t = A.MakeTime(date);
    const eq = A.Equator(name, t, this.observer(), false, true);   // J2000, topocentric
    return unit(eq.ra * 15, eq.dec);
  }

  // ---- frame loop ----------------------------------------------------------

  _astroTime() { return A.MakeTime(this.time); }
  _rot() { return this._rotCache || (this._rotCache = A.Rotation_EQJ_HOR(this._astroTime(), this.observer())); }
  _recompute() { this._rotCache = null; }

  _frame(now) {
    const dt = Math.min(0.25, (now - this._last) / 1000);
    this._last = now;
    if (this.rate !== 0) {
      this.time = new Date(this.time.getTime() + dt * 1000 * this.rate);
      this._recompute();
    }
    if (this.anim) {
      const { from, to, t0, ms } = this.anim;
      const k = Math.min(1, (now - t0) / ms);
      const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      this.view.az = (from.az + (to.az - from.az) * e + 360) % 360;
      this.view.alt = from.alt + (to.alt - from.alt) * e;
      // zoom out a little mid-flight on long slews, like a real planetarium
      this.view.fov = Math.exp(Math.log(from.fov) + (Math.log(to.fov) - Math.log(from.fov)) * e);
      if (k >= 1) this.anim = null;
    } else if (this.followTarget) {
      const h = this.horizontal(this.followTarget.vec());
      this.view.az = h.az; this.view.alt = Math.max(-80, Math.min(89.9, h.alt));
    }
    this._draw();
    requestAnimationFrame((t) => this._frame(t));
  }

  // ---- projection ----------------------------------------------------------

  _setupProjection() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = w; this.H = h;
    const az = this.view.az * DEG, alt = this.view.alt * DEG;
    // horizontal frame here: x east, y north, z up
    this.f = [Math.cos(alt) * Math.sin(az), Math.cos(alt) * Math.cos(az), Math.sin(alt)];
    this.r = [Math.cos(az), -Math.sin(az), 0];
    const f = this.f, r = this.r;
    this.u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
    this.scale = (Math.max(w, h) / 2) / (2 * Math.tan(this.view.fov * DEG / 4));
  }

  /** horizontal-frame vector (from the HOR rotation) -> screen [x, y] or null when behind */
  _projectHor(h, cull = -0.6) {
    const e = [-h.y, h.x, h.z];                             // to east/north/up
    const d = e[0] * this.f[0] + e[1] * this.f[1] + e[2] * this.f[2];
    if (d < cull) return null;
    const k = 2 * this.scale / (1 + d);
    const px = (e[0] * this.r[0] + e[1] * this.r[1] + e[2] * this.r[2]) * k;
    const py = (e[0] * this.u[0] + e[1] * this.u[1] + e[2] * this.u[2]) * k;
    return [this.W / 2 + px, this.H / 2 - py, h.z];
  }

  project(v, rot = this._rot()) {
    const t = this._astroTime();
    return this._projectHor(A.RotateVector(rot, new A.Vector(v[0], v[1], v[2], t)));
  }

  _projectAltAz(az, alt, cull) {
    const a = az * DEG, l = alt * DEG;
    return this._projectHor({ x: Math.cos(l) * Math.cos(a), y: -Math.cos(l) * Math.sin(a), z: Math.sin(l) }, cull);
  }

  _onScreen(p, m = 40) { return p && p[0] > -m && p[0] < this.W + m && p[1] > -m && p[1] < this.H + m; }

  // ---- drawing -------------------------------------------------------------

  _draw() {
    this._setupProjection();
    const ctx = this.ctx, W = this.W, H = this.H;
    const rot = this._rot();
    const t = this._astroTime();
    const obs = this.observer();
    const sunH = A.Horizon(t, obs, A.Equator("Sun", t, obs, true, true).ra, A.Equator("Sun", t, obs, true, true).dec);
    const atmo = this.overlays.has("atmosphere");
    // sky brightness 0 (night) .. 1 (day) from the Sun's altitude
    const day = atmo ? Math.max(0, Math.min(1, (sunH.altitude + 12) / 18)) : 0;
    const bg = [Math.round(4 + 70 * day), Math.round(8 + 120 * day), Math.round(20 + 170 * day)];
    ctx.fillStyle = `rgb(${bg})`;
    ctx.fillRect(0, 0, W, H);

    const P = (v) => this._projectHor(A.RotateVector(rot, new A.Vector(v[0], v[1], v[2], t)));
    const fov = this.view.fov;
    const magLimit = Math.min(6.5, 4.6 + Math.log10(120 / fov) * 2.2) - day * 5;

    if (this.overlays.has("grid")) this._drawGrid();
    if (this.overlays.has("ecliptic")) this._drawEcliptic(P);

    if (this.overlays.has("constellation lines")) {
      ctx.strokeStyle = `rgba(110,150,220,${0.45 - day * 0.3})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const c of this.cons) for (const seg of c.lines) {
        let prev = null;
        for (const v of seg) {
          const p = P(v);
          if (p && prev && Math.hypot(p[0] - prev[0], p[1] - prev[1]) < W) { ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); }
          prev = p;
        }
      }
      ctx.stroke();
    }

    for (const path of this.paths) this._drawPath(path, P);

    // stars
    const sizeK = Math.max(0.8, Math.min(2.2, Math.sqrt(60 / fov)));
    const labels = [];
    for (const s of this.stars) {
      if (s.mag > magLimit) break;                       // catalog sorted by magnitude
      const p = P(s.v);
      if (!this._onScreen(p, 4)) continue;
      const rad = Math.max(0.5, (magLimit - s.mag + 0.6) * 0.55 * sizeK);
      const alpha = Math.min(1, 0.35 + (magLimit - s.mag) * 0.3);
      ctx.fillStyle = `rgba(${s.rgb},${alpha})`;
      ctx.beginPath(); ctx.arc(p[0], p[1], rad, 0, 2 * Math.PI); ctx.fill();
      if (rad > 2.2) {                                     // soft glow on bright stars
        ctx.fillStyle = `rgba(${s.rgb},0.08)`;
        ctx.beginPath(); ctx.arc(p[0], p[1], rad * 3, 0, 2 * Math.PI); ctx.fill();
      }
      if (s.name && this.overlays.has("star names") && s.mag < magLimit - 2.2) labels.push([p, s.name, "#9fb7d8"]);
    }

    if (this.overlays.has("deep sky objects")) {
      ctx.strokeStyle = "rgba(140,220,170,0.7)";
      for (const d of this.dso) {
        if (d.mag > magLimit + 3) continue;
        const p = P(d.v);
        if (!this._onScreen(p)) continue;
        ctx.beginPath(); ctx.ellipse(p[0], p[1], 5, 3.5, 0, 0, 2 * Math.PI); ctx.stroke();
        if (fov < 70) labels.push([p, d.id, "rgba(140,220,170,0.85)"]);
      }
    }

    // solar system
    for (const name of BODIES) {
      const v = this.bodyVector(name);
      const p = P(v);
      if (!this._onScreen(p, 20)) continue;
      const [col, size] = BODY_STYLE[name];
      if (name === "Moon") this._drawMoon(p, size * sizeK);
      else {
        ctx.fillStyle = col;
        ctx.shadowColor = col; ctx.shadowBlur = name === "Sun" ? 30 : 8;
        ctx.beginPath(); ctx.arc(p[0], p[1], size * sizeK * (name === "Sun" ? 1 : 0.8), 0, 2 * Math.PI); ctx.fill();
        ctx.shadowBlur = 0;
      }
      if (this.overlays.has("planet names")) labels.push([p, name, "#ffd88a"]);
    }

    if (this.overlays.has("constellation names")) {
      ctx.font = "600 11px system-ui, sans-serif";
      ctx.textAlign = "center";
      for (const c of this.cons) {
        const p = P(c.v);
        if (!this._onScreen(p)) continue;
        ctx.fillStyle = `rgba(130,170,235,${0.55 - day * 0.35})`;
        ctx.fillText(c.name.toUpperCase(), p[0], p[1]);
      }
    }

    if (this.overlays.has("horizon")) this._drawGround(day);

    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "left";
    for (const [p, text, col] of labels) {
      if (this.overlays.has("horizon") && p[2] < -0.01 && this.view.alt >= -3) continue;
      ctx.fillStyle = col;
      ctx.fillText(text, p[0] + 7, p[1] - 6);
    }

    this._drawCardinals();
    if (this.highlight) this._drawHighlight(P);
  }

  _drawMoon(p, r) {
    const ctx = this.ctx;
    const phase = A.MoonPhase(this._astroTime());              // 0 new, 90 first quarter, 180 full
    const illum = (1 - Math.cos(phase * DEG)) / 2;
    ctx.fillStyle = "#2a2a2a";
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#f2f0e6";
    ctx.beginPath();
    const waxing = phase < 180;
    ctx.arc(p[0], p[1], r, -Math.PI / 2, Math.PI / 2, !waxing);
    ctx.ellipse(p[0], p[1], Math.abs(1 - 2 * illum) * r, r, 0, Math.PI / 2, -Math.PI / 2, illum > 0.5 ? !waxing : waxing);
    ctx.fill();
  }

  _drawGrid() {
    const ctx = this.ctx;
    ctx.strokeStyle = "rgba(120,140,170,0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let alt = -60; alt <= 80; alt += 10) {
      let prev = null;
      for (let az = 0; az <= 360; az += 3) {
        const p = this._projectAltAz(az, alt);
        if (p && prev) { ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); }
        prev = p;
      }
    }
    for (let az = 0; az < 360; az += 15) {
      let prev = null;
      for (let alt = -80; alt <= 90; alt += 3) {
        const p = this._projectAltAz(az, alt);
        if (p && prev) { ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); }
        prev = p;
      }
    }
    ctx.stroke();
  }

  _drawEcliptic(P) {
    const ctx = this.ctx;
    const eps = 23.4393 * DEG;
    ctx.strokeStyle = "rgba(230,190,90,0.45)";
    ctx.setLineDash([6, 5]);
    ctx.beginPath();
    let prev = null;
    for (let l = 0; l <= 360; l += 2) {
      const L = l * DEG;
      const p = P([Math.cos(L), Math.sin(L) * Math.cos(eps), Math.sin(L) * Math.sin(eps)]);
      if (p && prev && Math.hypot(p[0] - prev[0], p[1] - prev[1]) < this.W) { ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); }
      prev = p;
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  _drawPath(path, P) {
    const ctx = this.ctx;
    ctx.strokeStyle = "rgba(255,150,90,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let prev = null;
    for (const v of path.pts) {
      const p = P(v);
      if (p && prev) { ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); }
      prev = p;
    }
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillStyle = "rgba(255,180,120,0.95)";
    ctx.font = "11px system-ui, sans-serif";
    for (const tick of path.ticks) {
      const p = P(tick.v);
      if (!this._onScreen(p)) continue;
      ctx.beginPath(); ctx.arc(p[0], p[1], 2.5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillText(tick.label, p[0] + 5, p[1] + 12);
    }
  }

  _drawGround(day) {
    // The horizon is a circle under stereographic projection: fit it through
    // three projected points, then fill the side that contains the nadir.
    const ctx = this.ctx;
    const pts = [0, 120, 240].map((az) => this._projectAltAz(this.view.az + az + 0.5, 0, -0.9999)).filter(Boolean);
    // see-through ground when looking down, so objects below the horizon stay visible
    const opacity = this.view.alt < -3 ? 0.55 : 0.96;
    const ground = `rgba(${Math.round(12 + 30 * day)},${Math.round(16 + 35 * day)},${Math.round(14 + 20 * day)},${opacity})`;
    ctx.fillStyle = ground;
    const nadir = this._projectAltAz(0, -90, -0.9999);
    ctx.beginPath();
    if (pts.length === 3) {
      const [a, b, c] = pts;
      const d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
      if (Math.abs(d) > 1e-6) {
        const sa = a[0] ** 2 + a[1] ** 2, sb = b[0] ** 2 + b[1] ** 2, sc = c[0] ** 2 + c[1] ** 2;
        const ux = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d;
        const uy = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d;
        const rad = Math.hypot(a[0] - ux, a[1] - uy);
        const inside = nadir && Math.hypot(nadir[0] - ux, nadir[1] - uy) < rad;
        ctx.arc(ux, uy, rad, 0, 2 * Math.PI);
        if (!inside) { ctx.rect(this.W * 3, -this.H * 2, -this.W * 5, this.H * 5); }
        ctx.fill("evenodd");
        ctx.strokeStyle = "rgba(120,160,120,0.6)";
        ctx.beginPath(); ctx.arc(ux, uy, rad, 0, 2 * Math.PI); ctx.stroke();
        return;
      }
    }
    // looking almost exactly at the horizon: the horizon is a straight line
    const p = this._projectAltAz(this.view.az, 0);
    if (p) ctx.fillRect(0, p[1], this.W, this.H - p[1]);
  }

  _drawCardinals() {
    const ctx = this.ctx;
    ctx.font = "700 13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = "#e8b04a";
    for (const [az, n] of [[0, "N"], [45, "NE"], [90, "E"], [135, "SE"], [180, "S"], [225, "SW"], [270, "W"], [315, "NW"]]) {
      const p = this._projectAltAz(az, 0);
      if (this._onScreen(p)) ctx.fillText(n, p[0], p[1] + 16);
    }
  }

  _drawHighlight(P) {
    const { target, until } = this.highlight;
    const now = performance.now();
    if (now > until) { this.highlight = null; return; }
    const p = P(target.vec());
    if (!this._onScreen(p, 0)) return;
    const ctx = this.ctx;
    const pulse = 1 + 0.12 * Math.sin(now / 180);
    const r = 22 * pulse;
    ctx.strokeStyle = "rgba(255,210,120,0.95)";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI); ctx.stroke();
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      ctx.beginPath(); ctx.moveTo(p[0] + dx * r * 1.2, p[1] + dy * r * 1.2); ctx.lineTo(p[0] + dx * r * 1.7, p[1] + dy * r * 1.7); ctx.stroke();
    }
    ctx.font = "600 14px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillStyle = "#ffe0a0";
    ctx.fillText(target.label, p[0] + r * 1.4, p[1] - r * 0.9);
    ctx.lineWidth = 1;
  }

  // ---- mouse / touch -------------------------------------------------------

  _installInput() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, az: this.view.az, alt: this.view.alt };
      c.setPointerCapture(e.pointerId);
      this.anim = null; this.followTarget = null;
    });
    c.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const k = this.view.fov / Math.max(this.W, this.H);
      this.view.az = (drag.az - (e.clientX - drag.x) * k + 360) % 360;
      this.view.alt = Math.max(-85, Math.min(89.9, drag.alt + (e.clientY - drag.y) * k));
    });
    c.addEventListener("pointerup", () => { drag = null; });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.anim = null;
      this.view.fov = Math.max(0.5, Math.min(200, this.view.fov * Math.exp(e.deltaY * 0.0012)));
    }, { passive: false });
  }
}
