// The planetarium's tools: name resolution (Needle copies the user's words;
// everything it doesn't know - where Saturn is, where Tokyo is - lives here)
// and one executor per tool schema in data/tools.json.

import { BODIES } from "./sky.js";

const A = globalThis.Astronomy;
const DEG = Math.PI / 180;

// ---- name normalisation & fuzzy matching ------------------------------------

export function norm(s) {
  return String(s || "").toLowerCase()
    .replace(/[’']s\b/g, "").replace(/[^a-z0-9 ]+/g, " ")
    .replace(/\b(the|a|an|planet|star|constellation|please)\b/g, " ")
    .replace(/\s+/g, " ").trim();
}

function lev(a, b) {
  if (Math.abs(a.length - b.length) > 2) return 9;
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++)
    d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[a.length][b.length];
}

const unit = (raDeg, decDeg) => {
  const ra = raDeg * DEG, dec = decDeg * DEG;
  return [Math.cos(dec) * Math.cos(ra), Math.cos(dec) * Math.sin(ra), Math.sin(dec)];
};

// Distances for the info card, light years (planets are computed live).
const DISTANCE_LY = {
  sirius: 8.6, canopus: 310, arcturus: 37, vega: 25, capella: 43, rigel: 860, procyon: 11.5, betelgeuse: 550,
  altair: 16.7, aldebaran: 65, antares: 550, spica: 250, pollux: 34, fomalhaut: 25, deneb: 2600, regulus: 79,
  castor: 51, bellatrix: 250, alnilam: 2000, mizar: 83, alcor: 82, albireo: 430, polaris: 430, dubhe: 123,
  algol: 90, mira: 300, achernar: 139, hadar: 390, acrux: 320, shaula: 570, alioth: 81, enif: 690,
  alpheratz: 97, hamal: 66, thuban: 300, kochab: 130, alphard: 180, "rigil kentaurus": 4.37, mimosa: 280,
  M31: 2.5e6, M42: 1344, M45: 444, M44: 577, M1: 6500, M13: 22200, M51: 23e6, M104: 31e6, M57: 2570, M27: 1360,
  M8: 4100, M16: 7000, M20: 5200, M33: 2.7e6, M81: 11.8e6, M82: 12e6, M101: 21e6, M87: 53e6, M7: 980, M6: 1600,
  "Mel 25": 153, "NGC 869": 7500, "NGC 5139": 17000, LMC: 160000, SMC: 200000, "NGC 4755": 6400,
  "NGC 3372": 8500, "NGC 7293": 650, "B33": 1375, "NGC 2237": 5200, "NGC 7000": 2590, "NGC 253": 11.4e6, "NGC 5128": 12e6,
};
const DSO_TYPE = { gx: "galaxy", oc: "open star cluster", gc: "globular star cluster", en: "emission nebula",
  rn: "reflection nebula", pn: "planetary nebula", snr: "supernova remnant", dn: "dark nebula", ds: "double star",
  as: "asterism", ast: "asterism", sg: "galaxy", bn: "nebula", "en+oc": "nebula with star cluster" };
const KIND_TYPES = { galaxies: ["gx", "sg"], nebulae: ["en", "rn", "pn", "snr", "dn", "bn", "en+oc"], clusters: ["oc", "gc", "en+oc"] };

const ALIASES = {
  "north star": "polaris", "pole star": "polaris", "dog star": "sirius",
  "big dipper": "@UMa", plough: "@UMa", plow: "@UMa", "little dipper": "@UMi", "southern cross": "@Cru",
  "seven sisters": "M45", pleiades: "M45", beehive: "M44", "beehive cluster": "M44", praesepe: "M44",
  "orion belt": "alnilam", "orions belt": "alnilam", "summer triangle": "deneb", "evening star": "venus",
  "morning star": "venus", "red planet": "mars", luna: "moon", sol: "sun", "milky way center": "sgr a",
};

// ---- the resolver ----------------------------------------------------------

export class Resolver {
  constructor(sky, catalog, places) {
    this.sky = sky;
    this.index = new Map();         // normalised name -> target factory
    const add = (key, fn) => { const k = norm(key); if (k && !this.index.has(k)) this.index.set(k, fn); };

    for (const b of BODIES) add(b, () => this._body(b));
    for (const s of sky.stars) if (s.name) add(s.name, () => this._star(s));
    for (const c of sky.cons) { add(c.name, () => this._con(c)); add(c.abbr, () => this._con(c)); }
    for (const d of sky.dso) {
      const f = () => this._dso(d);
      add(d.id, f); add(d.id.replace(/^M(\d+)$/, "messier $1"), f); add(d.id.replace(/^M(\d+)$/, "m $1"), f);
      add(d.name, f);
      for (const al of d.aliases) add(al, f);
    }
    add("sgr a", () => ({ kind: "point", label: "Galactic Center", vec: () => unit(266.42, -29.0), info: "the center of the Milky Way" }));
    for (const [k, v] of Object.entries(ALIASES)) {
      const target = v.startsWith("@") ? sky.cons.find((c) => c.abbr === v.slice(1)) : null;
      add(k, target ? () => ({ ...this._con(target), label: k.replace(/\b\w/g, (m) => m.toUpperCase()) })
        : () => this.index.get(norm(v))?.());
    }
    this.keys = [...this.index.keys()];

    this.places = places;
    this.placeIndex = new Map();
    places.cities.forEach((c, i) => { const k = norm(c[0]); if (!this.placeIndex.has(k)) this.placeIndex.set(k, i); });
  }

  _body(name) {
    return { kind: "body", name, label: name === "Sun" || name === "Moon" ? `the ${name}` : name,
      vec: () => this.sky.bodyVector(name) };
  }
  _star(s) { return { kind: "star", name: s.name, label: s.name, star: s, vec: () => s.v }; }
  _con(c) { return { kind: "constellation", name: c.name, label: c.name, con: c, vec: () => c.v }; }
  _dso(d) {
    const label = d.name !== d.id ? `${d.name} (${d.id})` : d.id;
    return { kind: "dso", name: d.name, label, dso: d, vec: () => d.v };
  }

  object(spoken) {
    const q = norm(spoken).replace(/\bmessier (\d+)/, "m$1").replace(/\bm (\d+)\b/, "m$1").replace(/\bngc(\d)/, "ngc $1");
    if (!q) return null;
    const hit = this.index.get(q);
    if (hit) return hit();
    // "crab" -> "crab nebula", "andromeda galaxy" -> "andromeda galaxy", "jupiter s moons" -> "jupiter"
    const prefix = this.keys.filter((k) => k.length > q.length && k.startsWith(q + " "));
    if (prefix.length) return this.index.get(prefix.sort((a, b) => a.length - b.length)[0])();
    const words = q.split(" ");
    for (let n = words.length - 1; n >= 1; n--) {          // longest known leading phrase
      const sub = words.slice(0, n).join(" ");
      if (this.index.has(sub)) return this.index.get(sub)();
    }
    if (q.length >= 4) {
      let best = null, bestD = 3;
      for (const k of this.keys) { const d = lev(q, k); if (d < bestD) { best = k; bestD = d; } }
      if (best && bestD <= (q.length > 7 ? 2 : 1)) return this.index.get(best)();
    }
    return null;
  }

  place(spoken) {
    const q = norm(spoken).replace(/\b(city of|downtown)\b/g, "").trim();
    const P = this.places;
    const special = P.special[q] || P.special["the " + q];
    if (special) {
      const [label, lat, lon] = special;
      return { name: label, lat, lon, tz: `Etc/GMT${lon >= 7.5 ? "-" : "+"}${Math.abs(Math.round(lon / 15))}` };
    }
    const city = (i) => {
      const [name, region, country, lat, lon, , tz] = P.cities[i];
      const where = country === "United States" || country === "Canada" || country === "Australia" ? region : country;
      return { name: where && where !== name ? `${name}, ${where}` : name, lat, lon, tz };
    };
    const words = q.split(" ");
    // "sydney australia", "denver colorado", "portland oregon": city words + qualifier words
    for (let n = words.length; n >= 1; n--) {
      const head = words.slice(0, n).join(" "), rest = words.slice(n).join(" ");
      if (!rest) {
        if (this.placeIndex.has(head)) return city(this.placeIndex.get(head));
        if (P.regions[head] !== undefined) return city(P.regions[head]);
        continue;
      }
      const matches = P.cities.map((c, i) => [c, i]).filter(([c]) => norm(c[0]) === head
        && (norm(c[1]).includes(rest) || norm(c[2]).includes(rest) || rest.length <= 3 && norm(c[1]).startsWith(rest)));
      if (matches.length) return city(matches[0][1]);
    }
    if (q.length >= 4) {
      let best = null, bestD = 3;
      for (const k of this.placeIndex.keys()) { const d = lev(q, k); if (d < bestD) { best = k; bestD = d; } }
      if (best && bestD <= (q.length > 7 ? 2 : 1)) return city(this.placeIndex.get(best));
    }
    return null;
  }
}

// ---- time helpers -----------------------------------------------------------

/** Wall-clock fields in an IANA zone -> UTC Date. */
export function zonedToUtc(y, mo, d, h, mi, tz) {
  let guess = Date.UTC(y, mo - 1, d, h, mi);
  for (let i = 0; i < 2; i++) {
    const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
      timeZone: tz, hourCycle: "h23", year: "numeric", month: "numeric", day: "numeric", hour: "numeric", minute: "numeric",
    }).formatToParts(new Date(guess)).map((p) => [p.type, p.value]));
    const shown = Date.UTC(+parts.year, +parts.month - 1, +parts.day, +parts.hour, +parts.minute);
    guess += Date.UTC(y, mo - 1, d, h, mi) - shown;
  }
  return new Date(guess);
}

export function localParts(date, tz) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
    timeZone: tz, hourCycle: "h23", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", weekday: "short",
  }).formatToParts(date).map((p) => [p.type, p.value]));
  return parts;
}

export function fmtLocal(date, tz, withDate = true) {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: tz, hour: "numeric", minute: "2-digit", ...(withDate ? { month: "short", day: "numeric", year: "numeric" } : {}),
  }).format(date);
}

// ---- executors ---------------------------------------------------------------

const VIEWS = { "whole sky": 180, "naked eye": 90, binoculars: 7, telescope: 1.2 };
const DIRECTIONS = { north: 0, northeast: 45, east: 90, southeast: 135, south: 180, southwest: 225, west: 270, northwest: 315 };
const SPEEDS = { pause: 0, "real time": 1, "minute per second": 60, "hour per second": 3600, "day per second": 86400 };

export function makeExecutors(sky, resolver, ui) {
  const fovFor = (t) => t.kind === "constellation" ? 60 : t.kind === "dso" ? 8 : t.kind === "body" ? (t.name === "Moon" || t.name === "Sun" ? 6 : 3) : 20;
  const notFound = (what, name) => ({ ok: false, summary: `I don't know a ${what} called "${name}".` });
  const altText = (h) => h.alt > 0 ? `${h.alt.toFixed(0)}° up in the ${compass(h.az)}` : `below the horizon (${h.alt.toFixed(0)}°)`;

  function aim(target, fov) {
    const h = sky.horizontal(target.vec());
    sky.followTarget = null;
    sky.slewTo(h.az, Math.max(-80, Math.min(89.9, h.alt)), fov ?? Math.min(sky.view.fov, fovFor(target)));
    sky.highlight = { target, until: performance.now() + 6000 };
    return h;
  }

  return {
    show_object({ name }) {
      const t = resolver.object(name);
      if (!t) return notFound("sky object", name);
      const h = aim(t);
      return { ok: true, summary: `${t.label} is ${altText(h)}.` };
    },

    follow_object({ name }) {
      const t = resolver.object(name);
      if (!t) return notFound("sky object", name);
      aim(t);
      setTimeout(() => { sky.followTarget = t; }, 1450);
      return { ok: true, summary: `Following ${t.label}. Speed up time to watch it move.` };
    },

    set_location({ place }) {
      const p = resolver.place(place);
      if (!p) return { ok: false, summary: `I couldn't find a place called "${place}".` };
      sky.setPlace(p);
      ui.placeChanged?.(p);
      return { ok: true, summary: `Viewing from ${p.name}, where it's ${fmtLocal(sky.time, p.tz, false)}.` };
    },

    set_date_time({ date, time }) {
      const tz = sky.place.tz;
      const now = localParts(sky.time, tz);
      let [y, mo, d] = [+now.year, +now.month, +now.day];
      let [h, mi] = [+now.hour, +now.minute];
      if (date) { const m = /^(-?\d{1,4})-(\d{1,2})-(\d{1,2})/.exec(date); if (m) [y, mo, d] = [+m[1], +m[2], +m[3]]; }
      if (time) { const m = /^(\d{1,2}):(\d{2})/.exec(time); if (m) [h, mi] = [+m[1], +m[2]]; }
      if (!date && !time) return { ok: false, summary: "No date or time given." };
      const when = zonedToUtc(y, mo, d, h, mi, tz);
      sky.setTime(when);
      return { ok: true, summary: `Sky set to ${fmtLocal(when, tz)} in ${sky.place.name}.` };
    },

    reset_to_now() {
      sky.setTime(new Date());
      sky.setRate(1);
      return { ok: true, summary: "Back to the present, running in real time." };
    },

    set_time_speed({ speed, backwards }) {
      const r = (SPEEDS[speed] ?? 1) * (backwards ? -1 : 1);
      sky.setRate(r);
      return { ok: true, summary: r === 0 ? "Time paused." : `Time running ${backwards ? "backwards " : ""}at ${speed}.` };
    },

    zoom({ direction }) {
      sky.zoomBy(direction === "in" ? 0.5 : 2);
      return { ok: true, summary: `Zoomed ${direction}.` };
    },

    set_view({ view }) {
      sky.slewTo(sky.view.az, view === "whole sky" ? 89.9 : sky.view.alt, VIEWS[view] ?? 90, 900);
      return { ok: true, summary: `${view[0].toUpperCase() + view.slice(1)} view (${VIEWS[view]}° wide).` };
    },

    look_toward({ direction }) {
      sky.followTarget = null;
      if (direction === "up") sky.slewTo(sky.view.az, 89.9, Math.max(sky.view.fov, 90));
      else sky.slewTo(DIRECTIONS[direction] ?? 180, 25, Math.max(sky.view.fov, 70));
      return { ok: true, summary: direction === "up" ? "Looking straight up." : `Facing ${direction}.` };
    },

    show_overlay({ overlay }) { sky.setOverlay(overlay, true); ui.overlaysChanged?.(); return { ok: true, summary: `Showing ${overlay}.` }; },
    hide_overlay({ overlay }) { sky.setOverlay(overlay, false); ui.overlaysChanged?.(); return { ok: true, summary: `Hiding ${overlay}.` }; },

    show_path({ name, days }) {
      const t = resolver.object(name);
      if (!t) return notFound("sky object", name);
      if (t.kind !== "body" || t.name === "Sun") return { ok: false, summary: `${t.label} doesn't move against the stars.` };
      const n = Math.max(2, Math.min(3650, days || (t.name === "Moon" ? 27 : 120)));
      const steps = Math.min(600, n * (t.name === "Moon" ? 4 : 1));
      const pts = [], ticks = [];
      const tickEvery = Math.max(1, Math.round(n / 8));
      for (let i = 0; i <= steps; i++) {
        const when = new Date(sky.time.getTime() + (i / steps) * n * 86400e3);
        const g = A.GeoVector(t.name, A.MakeTime(when), true);
        const len = Math.hypot(g.x, g.y, g.z);
        const v = [g.x / len, g.y / len, g.z / len];
        pts.push(v);
        const day = (i / steps) * n;
        if (Math.abs(day / tickEvery - Math.round(day / tickEvery)) < (n / steps) / tickEvery / 2 + 1e-9)
          ticks.push({ v, label: new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: sky.place.tz }).format(when) });
      }
      sky.paths = [{ name: t.name, pts, ticks }];
      // aim at the middle of the track and open the view to fit it
      const mid = pts[Math.floor(pts.length / 2)];
      let span = 0;
      for (const p of pts) span = Math.max(span, Math.acos(Math.max(-1, Math.min(1, p[0] * mid[0] + p[1] * mid[1] + p[2] * mid[2]))) / DEG);
      const h = sky.horizontal(mid);
      sky.followTarget = null;
      sky.slewTo(h.az, Math.max(-80, Math.min(89.9, h.alt)), Math.min(180, Math.max(20, span * 2.6)));
      return { ok: true, summary: `Drew ${t.label}'s path over ${n} days.` };
    },

    object_info({ name }) {
      const t = resolver.object(name);
      if (!t) return notFound("sky object", name);
      const h = aim(t);
      const facts = [];
      const time = A.MakeTime(sky.time);
      if (t.kind === "body") {
        const dist = A.GeoVector(t.name, time, true);
        const au = Math.hypot(dist.x, dist.y, dist.z);
        facts.push(["Type", t.name === "Sun" ? "star" : t.name === "Moon" ? "Earth's moon" : t.name === "Pluto" ? "dwarf planet" : "planet"]);
        facts.push(["Distance", t.name === "Moon" ? `${Math.round(au * 149597870.7).toLocaleString()} km` : `${au.toFixed(2)} AU (${(au * 8.317).toFixed(1)} light-minutes)`]);
        if (t.name !== "Sun") {
          const ill = A.Illumination(t.name, time);
          facts.push(["Magnitude", ill.mag.toFixed(1)]);
          facts.push(["Lit", `${Math.round(ill.phase_fraction * 100)}%`]);
        }
      } else if (t.kind === "star") {
        const s = t.star;
        const con = sky.cons.find((c) => c.abbr === s.con);
        facts.push(["Type", "star"], ["Magnitude", s.mag.toFixed(2)]);
        if (con) facts.push(["Constellation", con.name]);
        const ly = DISTANCE_LY[norm(s.name)];
        if (ly) facts.push(["Distance", `${ly.toLocaleString()} light years`]);
      } else if (t.kind === "dso") {
        const d = t.dso;
        facts.push(["Type", DSO_TYPE[d.type] || d.type], ["Catalog", [d.id, ...d.aliases].join(", ")], ["Magnitude", String(d.mag)]);
        const con = A.Constellation(d.ra / 15, d.dec);
        facts.push(["Constellation", con.name]);
        const ly = DISTANCE_LY[d.id];
        if (ly) facts.push(["Distance", ly >= 1e6 ? `${(ly / 1e6).toFixed(1)} million light years` : `${ly.toLocaleString()} light years`]);
      } else if (t.kind === "constellation") {
        const stars = sky.stars.filter((s) => s.con === t.con.abbr && s.name).slice(0, 4).map((s) => s.name);
        facts.push(["Type", "constellation"]);
        if (stars.length) facts.push(["Brightest named stars", stars.join(", ")]);
      }
      facts.push(["Position now", altText(h)]);
      ui.showCard?.({ title: t.label, facts });
      return { ok: true, summary: `${t.label}: ${facts.slice(0, 3).map(([k, v]) => `${k.toLowerCase()} ${v}`).join("; ")}.` };
    },

    rise_set_times({ name }) {
      const t = resolver.object(name);
      if (!t) return notFound("sky object", name);
      const obs = sky.observer();
      const start = A.MakeTime(new Date(sky.time.getTime() - 12 * 3600e3));
      let body = t.name;
      if (t.kind !== "body") {
        const v = t.vec();
        const ra = ((Math.atan2(v[1], v[0]) / DEG + 360) % 360) / 15, dec = Math.asin(v[2]) / DEG;
        A.DefineStar(A.Body.Star1, ra, dec, 1000);
        body = A.Body.Star1;
      }
      const tz = sky.place.tz;
      const find = (dir) => { try { return A.SearchRiseSet(body, obs, dir, start, 2); } catch { return null; } };
      const rise = find(+1), set = find(-1);
      let transit = null;
      try { transit = A.SearchHourAngle(body, obs, 0, start, +1); } catch { /* circumpolar edge cases */ }
      const facts = [];
      if (!rise && !set) {
        const h = sky.horizontal(t.vec());
        facts.push(["", h.alt > 0 ? "Never sets from here (circumpolar)" : "Never rises from here"]);
      } else {
        if (rise) facts.push(["Rises", fmtLocal(rise.date, tz)]);
        if (transit) facts.push(["Highest", `${fmtLocal(transit.time.date, tz)} at ${transit.hor.altitude.toFixed(0)}°`]);
        if (set) facts.push(["Sets", fmtLocal(set.date, tz)]);
      }
      ui.showCard?.({ title: `${t.label} from ${sky.place.name}`, facts });
      aim(t);
      return { ok: true, summary: facts.map(([k, v]) => `${k} ${v}`.trim()).join("; ") + "." };
    },

    list_visible({ kind, count }) {
      const n = Math.max(1, Math.min(30, count || 8));
      const rot = sky._rot();
      let items = [];
      const up = (v) => sky.horizontal(v, rot);
      if (kind === "planets") {
        for (const b of BODIES) {
          if (b === "Sun") continue;
          const h = up(sky.bodyVector(b));
          if (h.alt > 0) items.push({ name: b, mag: A.Illumination(b, A.MakeTime(sky.time)).mag, h });
        }
      } else if (kind === "stars") {
        for (const s of sky.stars) { if (!s.name) continue; const h = up(s.v); if (h.alt > 5) items.push({ name: s.name, mag: s.mag, h }); if (items.length >= n) break; }
      } else if (kind === "constellations") {
        for (const c of sky.cons) { const h = up(c.v); if (h.alt > 15) items.push({ name: c.name, mag: -h.alt, h }); }
      } else {
        const types = KIND_TYPES[kind] || [];
        for (const d of sky.dso) if (types.includes(d.type)) { const h = up(d.v); if (h.alt > 10) items.push({ name: d.name !== d.id ? `${d.name} (${d.id})` : d.id, mag: d.mag, h }); }
      }
      items.sort((a, b) => a.mag - b.mag);
      items = items.slice(0, n);
      ui.showCard?.({
        title: `${kind[0].toUpperCase() + kind.slice(1)} up now from ${sky.place.name}`,
        facts: items.length ? items.map((it) => [it.name, `${it.h.alt.toFixed(0)}° ${compass(it.h.az)}`]) : [["", "None above the horizon right now."]],
        clickable: true,
      });
      return { ok: true, summary: items.length ? `${items.length} ${kind} up: ${items.map((i) => i.name).join(", ")}.` : `No ${kind} above the horizon.` };
    },
  };
}

export function compass(az) {
  return ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"][Math.round(az / 45) % 8];
}
