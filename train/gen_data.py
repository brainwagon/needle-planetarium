"""Generate planetarium fine-tuning data for Needle 3.

Every example shows the model 5 tools, the way the engine's retrieval does at
run time: the right tool(s) plus distractors, some of them deliberately close
(show/hide, show_object/object_info).  Every argument value is a span of the
query (or an enum value the query names), per Needle's grounding contract.

With --all-tools every example shows all 15 tools instead: a locally tuned model
has no retrieval head, so that is what it sees at run time.

Usage: python train/gen_data.py --n 6000 --out train/data.jsonl [--all-tools]
"""
import argparse
import datetime as dt
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = {t["name"]: t for t in json.loads((ROOT / "spike" / "tools2.json").read_text())}

# Near-miss pairs: when one is the answer, the other is a likely distractor.
CONFUSABLE = {
    "show_object": ["object_info", "follow_object", "set_location", "show_path"],
    "follow_object": ["show_object", "show_path", "set_time_speed"],
    "set_location": ["show_object", "object_info", "look_toward"],
    "set_date_time": ["reset_to_now", "set_time_speed", "rise_set_times"],
    "reset_to_now": ["set_date_time", "set_time_speed"],
    "set_time_speed": ["set_date_time", "reset_to_now", "follow_object"],
    "zoom": ["set_view", "look_toward"],
    "set_view": ["zoom", "show_overlay"],
    "look_toward": ["zoom", "show_object", "set_view"],
    "show_overlay": ["hide_overlay", "list_visible", "object_info"],
    "hide_overlay": ["show_overlay", "set_view"],
    "show_path": ["follow_object", "show_object", "rise_set_times"],
    "object_info": ["show_object", "rise_set_times", "list_visible"],
    "rise_set_times": ["object_info", "set_date_time", "list_visible"],
    "list_visible": ["show_object", "object_info", "show_overlay"],
}

PLANETS = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
MOVERS = PLANETS + ["the Moon", "the moon", "moon", "the Sun", "the sun", "Ceres", "Vesta"]
STARS = ["Sirius", "Canopus", "Arcturus", "Vega", "Capella", "Rigel", "Procyon", "Betelgeuse",
         "Altair", "Aldebaran", "Antares", "Spica", "Pollux", "Fomalhaut", "Deneb", "Regulus",
         "Castor", "Bellatrix", "Alnilam", "Mizar", "Alcor", "Albireo", "Polaris", "the North Star",
         "Dubhe", "Algol", "Mira", "Achernar", "Hadar", "Acrux", "Shaula", "Alioth", "Rasalhague",
         "Enif", "Markab", "Alpheratz", "Hamal", "Thuban", "Eltanin", "Kochab", "Alphard"]
CONSTELLATIONS = ["Orion", "Ursa Major", "Ursa Minor", "Cassiopeia", "Cygnus", "Lyra", "Scorpius",
                  "Sagittarius", "Leo", "Gemini", "Taurus", "Andromeda", "Perseus", "Pegasus",
                  "Aquila", "Bootes", "Virgo", "Canis Major", "Crux", "the Southern Cross",
                  "the Big Dipper", "the Little Dipper", "Draco", "Hercules", "Auriga", "Cepheus",
                  "Aquarius", "Pisces", "Capricornus", "Libra", "Ophiuchus", "Centaurus",
                  "Corona Borealis", "Cancer", "Aries", "Delphinus", "Lepus", "Monoceros"]
DSO = ["the Andromeda Galaxy", "Andromeda Galaxy", "the Orion Nebula", "the Pleiades", "Pleiades",
       "the Hyades", "the Beehive Cluster", "the Whirlpool Galaxy", "the Sombrero Galaxy",
       "the Lagoon Nebula", "the Eagle Nebula", "the Dumbbell Nebula", "the Helix Nebula",
       "the Triangulum Galaxy", "the Pinwheel Galaxy", "the Double Cluster", "Omega Centauri",
       "the Large Magellanic Cloud", "the Small Magellanic Cloud", "the Horsehead Nebula",
       "the Trifid Nebula", "the Owl Nebula", "Bode's Galaxy", "the Cigar Galaxy",
       "the Wild Duck Cluster", "the Butterfly Cluster", "the Veil Nebula", "the Rosette Nebula",
       "the Hercules Cluster", "the Jewel Box", "the Carina Nebula", "the Tarantula Nebula"]
MESSIER = [f"M{n}" for n in range(1, 111)] + [f"NGC {n}" for n in (253, 869, 884, 891, 2237, 3372, 4565, 5128, 6543, 7000, 7293, 7331)]
OBJECTS = PLANETS * 3 + MOVERS + STARS + CONSTELLATIONS + DSO + MESSIER

PLACES = ["London", "Paris", "Berlin", "Madrid", "Rome", "Oslo", "Stockholm", "Helsinki", "Moscow",
          "Cairo", "Nairobi", "Lagos", "Johannesburg", "Dubai", "Mumbai", "Delhi", "Bangkok",
          "Singapore", "Beijing", "Shanghai", "Seoul", "Hong Kong", "Manila", "Jakarta", "Perth",
          "Melbourne", "Brisbane", "Auckland", "Wellington", "Honolulu", "Anchorage", "Vancouver",
          "Seattle", "Portland", "San Francisco", "Los Angeles", "San Diego", "Phoenix", "Las Vegas",
          "Salt Lake City", "Chicago", "Houston", "Dallas", "Miami", "Atlanta", "Boston", "New York",
          "Toronto", "Montreal", "Mexico City", "Bogota", "Lima", "Santiago", "Buenos Aires",
          "Sao Paulo", "Rio de Janeiro", "Havana", "Dublin", "Edinburgh", "Lisbon", "Athens",
          "Istanbul", "Tehran", "Karachi", "Kathmandu", "Ulaanbaatar", "Tromso", "Fairbanks",
          "the North Pole", "the Equator", "Mauna Kea", "the Atacama Desert", "Antarctica",
          "Tasmania", "Iceland", "Greenland", "Hawaii", "Alaska", "Patagonia", "Madagascar",
          "Austin Texas", "Tucson Arizona", "Boulder Colorado", "Ithaca New York",
          "Christchurch New Zealand", "Kyoto Japan", "Lyon France", "Munich Germany"]
OVERLAYS = {
    "constellation lines": ["constellation lines", "the constellation lines", "constellation figures", "the stick figures"],
    "constellation names": ["constellation names", "the constellation names", "constellation labels"],
    "star names": ["star names", "the star names", "star labels", "the star labels"],
    "planet names": ["planet names", "the planet labels", "planet labels"],
    "deep sky objects": ["deep sky objects", "the deep sky objects", "galaxies and nebulae"],
    "grid": ["grid", "the grid", "the coordinate grid", "the gridlines"],
    "ecliptic": ["ecliptic", "the ecliptic", "the ecliptic line"],
    "horizon": ["horizon", "the horizon", "the ground"],
    "atmosphere": ["atmosphere", "the atmosphere", "the sky glow"],
}
DIRS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                8: "eight", 9: "nine", 10: "ten", 12: "twelve", 20: "twenty"}


def pick(xs):
    return random.choice(xs)


def cap(s):
    return s[0].upper() + s[1:] if random.random() < 0.35 else s


def bare(name):
    """'the Moon' -> 'Moon' sometimes, so both forms appear as spans."""
    return name


# Each intent returns (query, [answers], reasoning).  Queries keep the argument
# text verbatim so the argument is a span of the query.

def i_show():
    o = pick(OBJECTS)
    t = pick(["show me {o}", "show {o}", "where is {o}", "where's {o}", "find {o}", "find {o} for me",
              "point at {o}", "point to {o}", "go to {o}", "take me to {o}", "center on {o}",
              "look at {o}", "can you show me {o}", "I want to see {o}", "bring up {o}",
              "aim at {o}", "let's see {o}", "{o} please", "pull up {o}", "where can I find {o}",
              "navigate to {o}", "jump to {o}", "display {o}", "locate {o}", "zoom to {o}"])
    return t.format(o=o), [("show_object", {"name": o})], f"'{o}' -> name"


def i_follow():
    o = pick(MOVERS)
    t = pick(["track {o}", "follow {o}", "keep {o} centered", "keep {o} in the middle",
              "keep {o} in view", "lock onto {o}", "stay on {o}", "follow {o} across the sky",
              "track {o} as it moves", "keep the view on {o}", "chase {o}", "keep following {o}",
              "lock the view on {o}", "keep {o} in the center of the screen"])
    return t.format(o=o), [("follow_object", {"name": o})], f"'{o}' -> name"


def i_location():
    p = pick(PLACES)
    t = pick(["view from {p}", "show the sky from {p}", "what does the sky look like from {p}",
              "move me to {p}", "set my location to {p}", "change location to {p}", "I'm in {p}",
              "I am in {p}", "go to {p}", "take me to {p}", "show the sky over {p}",
              "what can I see from {p}", "put me in {p}", "observe from {p}", "pretend I'm in {p}",
              "switch to {p}", "the sky above {p}", "teleport me to {p}", "relocate to {p}"])
    return t.format(p=p), [("set_location", {"place": p})], f"'{p}' -> place"


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def fmt_time(h, m):
    styles = []
    h12 = h % 12 or 12
    ap = "am" if h < 12 else "pm"
    if m == 0:
        styles += [f"{h12}{ap}", f"{h12} {ap}", f"{h12} o'clock" if 6 <= h12 <= 11 and False else f"{h12}{ap}"]
        if h == 0:
            styles += ["midnight"]
        if h == 12:
            styles += ["noon", "midday"]
    else:
        styles += [f"{h12}:{m:02d}{ap}", f"{h12}:{m:02d} {ap}"]
    styles += [f"{h:02d}:{m:02d}"]
    return pick(styles)


def fmt_date(d):
    styles = [f"{MONTHS[d.month - 1]} {d.day} {d.year}", f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
              f"{d.day} {MONTHS[d.month - 1]} {d.year}", d.isoformat(),
              f"{MONTHS[d.month - 1][:3]} {d.day} {d.year}"]
    return pick(styles)


def i_datetime(now):
    d = dt.date(random.randint(1900, 2100), random.randint(1, 12), random.randint(1, 28))
    h, m = random.randint(0, 23), pick([0, 0, 0, 15, 30, 45, 10, 5])
    ds, ts = fmt_date(d), fmt_time(h, m)
    kind = random.random()
    if kind < 0.6:
        t = pick(["show the sky on {d} at {t}", "set the time to {t} on {d}", "go to {d} at {t}",
                  "what did the sky look like on {d} at {t}", "jump to {d} {t}", "{d} at {t}",
                  "set date to {d} and time to {t}", "take me to {t} on {d}",
                  "what will the sky look like on {d} at {t}", "change the time to {d}, {t}"])
        return (t.format(d=ds, t=ts), [("set_date_time", {"date": d.isoformat(), "time": f"{h:02d}:{m:02d}"})],
                f"'{ds}' -> date; '{ts}' -> time")
    if kind < 0.8:
        t = pick(["show the sky on {d}", "go to {d}", "set the date to {d}", "what did the sky look like on {d}",
                  "jump to {d}"])
        return t.format(d=ds), [("set_date_time", {"date": d.isoformat()})], f"'{ds}' -> date"
    # time only, today
    t = pick(["set the time to {t}", "show me {t} tonight", "what will it look like at {t}",
              "go to {t}", "jump to {t}", "show the sky at {t}"])
    return t.format(t=ts), [("set_date_time", {"time": f"{h:02d}:{m:02d}"})], f"'{ts}' -> time"


def i_now():
    t = pick(["back to the present", "reset to now", "go back to now", "current time", "return to the present",
              "show the sky right now", "back to today", "go to the current time", "reset the clock",
              "now please", "return to now", "jump back to the present day", "reset time"])
    return t, [("reset_to_now", {})], "present -> reset"


SPEEDS = {
    "pause": ["pause the clock", "pause time", "stop the clock", "freeze time", "stop time", "pause",
              "hold the time", "stop the animation"],
    "real time": ["real time", "go back to real time", "normal speed", "run at normal speed",
                  "set time to real time", "play at real speed"],
    "minute per second": ["a minute per second", "one minute per second", "a minute every second",
                          "1 minute per second"],
    "hour per second": ["an hour per second", "one hour per second", "an hour every second",
                        "1 hour per second"],
    "day per second": ["a day per second", "one day per second", "a day every second", "1 day per second"],
}


def i_speed():
    s = pick(list(SPEEDS))
    phrase = pick(SPEEDS[s])
    back = s not in ("pause",) and random.random() < 0.3
    if s in ("minute per second", "hour per second", "day per second"):
        if back:
            t = pick(["run time backwards at {p}", "rewind at {p}", "go backwards {p}",
                      "reverse time, {p}", "run backwards {p}", "rewind time at {p}"])
        else:
            t = pick(["speed up time to {p}", "run time at {p}", "set time speed to {p}", "fast forward {p}",
                      "make time go {p}", "speed time up, {p}", "{p}", "animate at {p}"])
        q = t.format(p=phrase)
    elif s == "real time" and back:
        q = pick(["run time backwards in real time", "rewind in real time"])
    else:
        back = False
        q = phrase
    args = {"speed": s}
    if back:
        args["backwards"] = True
    return q, [("set_time_speed", args)], f"'{phrase}' -> speed" + ("; backwards -> backwards true" if back else "")


def i_zoom():
    d = pick(["in", "out"])
    q = pick({"in": ["zoom in", "zoom in more", "closer", "get closer", "magnify", "zoom in a bit",
                     "zoom in please", "enlarge the view", "move in closer"],
              "out": ["zoom out", "zoom out more", "pull back", "pull back a bit", "wider", "zoom out a little",
                      "back out", "zoom out please", "show more of the sky"]}[d])
    return q, [("zoom", {"direction": d})], f"zoom {d} -> direction"


def i_view():
    v = pick(["whole sky", "naked eye", "binoculars", "telescope"])
    q = pick({"whole sky": ["show the whole sky", "whole sky view", "all-sky view", "show the entire sky",
                            "fisheye view of the whole sky"],
              "naked eye": ["naked eye view", "show what I'd see with the naked eye", "naked eye please",
                            "go to naked eye field of view"],
              "binoculars": ["binocular view", "binoculars view", "show it like binoculars",
                             "give me a binoculars view", "as if through binoculars"],
              "telescope": ["telescope view", "through a telescope", "telescope field of view",
                            "show it like a telescope", "switch to telescope view"]}[v])
    return q, [("set_view", {"view": v})], f"'{v}' -> view"


def i_look():
    d = pick(DIRS + ["up"] * 2)
    if d == "up":
        q = pick(["look up", "look straight up", "point straight up", "look overhead",
                  "show me straight overhead", "face up"])
    else:
        q = pick(["face {d}", "look {d}", "turn {d}", "look to the {d}", "point the view {d}",
                  "turn to face {d}", "rotate to {d}", "look toward the {d}", "show the {d} sky",
                  "turn around and face {d}", "what's in the {d}"]).format(d=d)
    return q, [("look_toward", {"direction": d})], f"'{d}' -> direction"


def i_overlay(on):
    o = pick(list(OVERLAYS))
    phrase = pick(OVERLAYS[o])
    if on:
        q = pick(["turn on {p}", "show {p}", "show me {p}", "display {p}", "enable {p}", "add {p}",
                  "put {p} on", "turn {p} on", "switch on {p}", "I want {p}", "draw {p}"]).format(p=phrase)
        return q, [("show_overlay", {"overlay": o})], f"'{phrase}' -> overlay"
    q = pick(["turn off {p}", "hide {p}", "remove {p}", "get rid of {p}", "disable {p}", "switch off {p}",
              "turn {p} off", "no more {p}", "take away {p}", "lose {p}", "clear {p}"]).format(p=phrase)
    return q, [("hide_overlay", {"overlay": o})], f"'{phrase}' -> overlay"


def i_path():
    o = pick(MOVERS)
    if random.random() < 0.75:
        n = pick([7, 10, 14, 30, 45, 60, 90, 100, 120, 180, 200, 365, 500, 730])
        ns = str(n) if random.random() < 0.8 or n not in NUMBER_WORDS else NUMBER_WORDS[n]
        poss = o + ("'" if o.endswith("s") else "'s")
        q = pick([f"draw the path of {o} over {ns} days", f"show {poss} path for the next {ns} days",
                  f"plot {poss} motion over {ns} days", f"trace {o} for {ns} days",
                  f"show where {o} goes over {ns} days", f"draw {poss} track for {ns} days"])
        return q, [("show_path", {"name": o, "days": n})], f"'{o}' -> name; '{ns}' -> days"
    q = pick([f"draw the path of {o}", f"show the path of {o}", f"trace {o}'s motion", f"plot the track of {o}"])
    return q, [("show_path", {"name": o})], f"'{o}' -> name"


def i_info():
    o = pick(OBJECTS)
    q = pick(["tell me about {o}", "what is {o}", "how far away is {o}", "how far is {o}", "info on {o}",
              "how big is {o}", "how bright is {o}", "give me facts about {o}", "describe {o}",
              "what kind of object is {o}", "details about {o}", "what do you know about {o}",
              "explain {o}"]).format(o=o)
    return q, [("object_info", {"name": o})], f"'{o}' -> name"


def i_riseset():
    o = pick(MOVERS + STARS[:12] + CONSTELLATIONS[:8])
    q = pick(["when does {o} rise", "when will {o} set", "what time does {o} rise", "when does {o} set tonight",
              "rise and set times for {o}", "when is {o} up", "when can I see {o} rise",
              "when does {o} come up", "when does {o} go down", "what time will {o} set"]).format(o=o)
    if random.random() < 0.15:
        sun = pick(["sunrise", "sunset"])
        q = pick([f"when is {sun}", f"what time is {sun}", f"{sun} time"])
        return q, [("rise_set_times", {"name": "sun"})], f"'{sun}' -> name sun"
    return q, [("rise_set_times", {"name": o})], f"'{o}' -> name"


KINDS = {"planets": ["planets"], "stars": ["stars", "bright stars"], "constellations": ["constellations"],
         "galaxies": ["galaxies"], "nebulae": ["nebulae", "nebulas"], "clusters": ["clusters", "star clusters"]}


def i_list():
    k = pick(list(KINDS))
    phrase = pick(KINDS[k])
    if random.random() < 0.4:
        n = random.randint(2, 12)
        ns = str(n) if random.random() < 0.6 or n not in NUMBER_WORDS else NUMBER_WORDS[n]
        q = pick([f"list the {ns} brightest {phrase}", f"name {ns} {phrase} I can see", f"show {ns} visible {phrase}",
                  f"top {ns} {phrase} up now", f"give me {ns} {phrase} that are visible"])
        return q, [("list_visible", {"kind": k, "count": n})], f"'{phrase}' -> kind; '{ns}' -> count"
    q = pick([f"which {phrase} are up", f"which {phrase} are visible", f"what {phrase} can I see",
              f"what {phrase} are up tonight", f"list visible {phrase}", f"list the {phrase}",
              f"any {phrase} up right now", f"what {phrase} are out"])
    return q, [("list_visible", {"kind": k})], f"'{phrase}' -> kind"


OFFTOPIC = ["what's the weather tomorrow", "set an alarm for 6am", "play some jazz", "call mom",
            "how tall is Mount Everest", "send a text to Sam", "what's 15 times 23", "order a pizza",
            "turn on the kitchen lights", "write me a haiku", "translate hello into French",
            "how do I bake bread", "what's the capital of Peru", "remind me to buy milk", "tell me a joke",
            "who won the game last night", "what's my battery level", "open the garage", "book a flight to Rome",
            "what's the stock price of Apple", "start a timer for 10 minutes", "how many calories in an apple",
            "volume up", "lock the front door", "what time is it in Tokyo", "convert 5 miles to km",
            "who wrote Hamlet", "spell astronomy", "good morning", "thanks", "never mind", "hello",
            "what's your name", "write a poem about the moon", "sing a song", "how old is the universe",
            "is it going to rain", "navigate to the nearest gas station", "email my boss",
            "what's the best telescope to buy", "how do black holes form", "why is the sky blue"]

SINGLE = [(i_show, 14), (i_follow, 6), (i_location, 8), (i_datetime, 9), (i_now, 3), (i_speed, 6),
          (i_zoom, 4), (i_view, 4), (i_look, 5), (lambda: i_overlay(True), 6), (lambda: i_overlay(False), 6),
          (i_path, 5), (i_info, 7), (i_riseset, 6), (i_list, 5)]


def single(now):
    fns, ws = zip(*SINGLE)
    fn = random.choices(fns, ws)[0]
    return fn(now) if fn is i_datetime else fn()


JOINERS = [" and ", " then ", ", then ", " and then ", ", "]


def multi(now):
    a, b = single(now), single(now)
    if a[1][0][0] == b[1][0][0]:
        return a
    second = b[0]
    # "show me Saturn from Tokyo" style
    if a[1][0][0] == "show_object" and b[1][0][0] == "set_location" and random.random() < 0.5:
        o, p = a[1][0][1]["name"], b[1][0][1]["place"]
        q = pick([f"show me {o} from {p}", f"show {o} as seen from {p}", f"where is {o} from {p}",
                  f"find {o} from {p}"])
        return q, b[1] + a[1], f"'{p}' -> place; '{o}' -> name"
    return a[0] + pick(JOINERS) + second, a[1] + b[1], a[2] + "; " + b[2]


def tool_subset(answer_names):
    names = list(dict.fromkeys(answer_names))
    near = [n for a in names for n in CONFUSABLE.get(a, []) if n not in names]
    random.shuffle(near)
    for n in near:
        if len(names) >= 5 or random.random() < 0.3:
            continue
        names.append(n)
    rest = [n for n in TOOLS if n not in names]
    random.shuffle(rest)
    names += rest[: 5 - len(names)]
    random.shuffle(names)
    return [TOOLS[n] for n in names]


def date_fact(now):
    return f"date: {now:%Y-%m-%d %a %H:%M}; app: planetarium"


def make(n, seed, exclude, all_tools=False):
    random.seed(seed)
    seen, rows = set(), []
    while len(rows) < n:
        now = dt.datetime(2026, random.randint(1, 12), random.randint(1, 28), random.randint(0, 23), random.randint(0, 59))
        r = random.random()
        if r < 0.08:
            q, answers, reasoning = pick(OFFTOPIC), [], "no tool fits"
        elif r < 0.20:
            q, answers, reasoning = multi(now)
        else:
            q, answers, reasoning = single(now)
        q = cap(q)
        if random.random() < 0.2:
            q += pick(["?", ".", " please", " thanks", "!"]) if not q.endswith("?") else ""
        if q.lower() in exclude or q in seen:
            continue
        seen.add(q)
        names = [a[0] for a in answers] or random.sample(list(TOOLS), 2)
        subset = tool_subset(names)               # drawn either way, so --all-tools keeps the same queries
        rows.append({
            "system": date_fact(now),
            "query": q,
            "tools": list(TOOLS.values()) if all_tools else subset,
            "reasoning": reasoning,
            "answers": [{"name": nm, "arguments": args} for nm, args in answers],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "train" / "data.jsonl"))
    ap.add_argument("--all-tools", action="store_true", help="show all 15 tools in every example")
    args = ap.parse_args()
    exclude = set()
    for f in ("cases2.json", "cases3.json"):
        exclude |= {c["q"].lower().rstrip("?.!") for c in json.loads((ROOT / "spike" / f).read_text())}
        exclude |= {c["q"].lower() for c in json.loads((ROOT / "spike" / f).read_text())}
    rows = make(args.n, args.seed, exclude, args.all_tools)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(r["answers"][0]["name"] if r["answers"] else "(refusal)" for r in rows)
    multi_n = sum(len(r["answers"]) > 1 for r in rows)
    print(f"wrote {len(rows)} rows to {args.out}; multi-call {multi_n}")
    for k, v in c.most_common():
        print(f"  {k:<16} {v}")


if __name__ == "__main__":
    main()
