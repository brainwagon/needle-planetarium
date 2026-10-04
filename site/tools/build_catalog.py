"""Pack the d3-celestial catalogs into one compact site/data/sky.json.

stars:   [ra_deg, dec_deg, mag, bv, name|null, constellation]   (mag <= 6)
cons:    {abbr: {name, ra, dec, lines: [[[ra, dec], ...], ...]}}
dso:     [{id, name, aliases, type, mag, ra, dec}]   Messier + a few famous extras
aliases: lowercased spoken name -> canonical key, for the object resolver
"""
import json
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
RAW = SITE.parent / "vendor" / "raw"   # d3-celestial data/*.json (BSD-3, Olaf Frohn)
DATA = SITE / "data"


def ra360(ra):
    return round(ra % 360, 4)


def main():
    names = json.loads((RAW / "starnames.json").read_text())
    stars = []
    for f in json.loads((RAW / "stars.6.json").read_text())["features"]:
        ra, dec = f["geometry"]["coordinates"]
        p = f["properties"]
        n = names.get(str(f["id"]), {})
        bv = float(p["bv"]) if p.get("bv") not in (None, "") else 0.6
        stars.append([ra360(ra), round(dec, 4), p["mag"], round(bv, 2), n.get("name") or None, n.get("c") or None])
    stars.sort(key=lambda s: s[2])

    cons = {}
    for f in json.loads((RAW / "constellations.json").read_text())["features"]:
        ra, dec = f["geometry"]["coordinates"]
        cons[f["id"]] = {"name": f["properties"]["name"], "ra": ra360(ra), "dec": dec, "lines": []}
    for f in json.loads((RAW / "constellations.lines.json").read_text())["features"]:
        cons[f["id"]]["lines"] = [[[ra360(a), round(d, 3)] for a, d in seg] for seg in f["geometry"]["coordinates"]]

    dso = []
    for f in json.loads((RAW / "messier.json").read_text())["features"]:
        ra, dec = f["geometry"]["coordinates"]
        p = f["properties"]
        aliases = [p["desig"]] if p.get("desig") else []
        dso.append({"id": f["id"], "name": p.get("alt") or f["id"], "aliases": aliases,
                    "type": p["type"], "mag": p["mag"], "ra": ra360(ra), "dec": dec})
    extras = [  # famous non-Messier objects people ask for by name
        ("NGC 869", "Double Cluster", ["NGC 884", "h and chi Persei"], "oc", 3.7, 35.0, 57.13),
        ("NGC 5139", "Omega Centauri", [], "gc", 3.9, 201.70, -47.48),
        ("LMC", "Large Magellanic Cloud", [], "gx", 0.9, 80.89, -69.76),
        ("SMC", "Small Magellanic Cloud", [], "gx", 2.7, 13.19, -72.83),
        ("NGC 4755", "Jewel Box", ["Kappa Crucis Cluster"], "oc", 4.2, 193.4, -60.37),
        ("NGC 3372", "Carina Nebula", ["Eta Carinae Nebula"], "en", 1.0, 161.26, -59.87),
        ("NGC 2070", "Tarantula Nebula", [], "en", 8.0, 84.68, -69.1),
        ("B33", "Horsehead Nebula", [], "dn", 6.8, 85.25, -2.46),
        ("NGC 7293", "Helix Nebula", [], "pn", 7.6, 337.41, -20.84),
        ("NGC 6960", "Veil Nebula", ["Cygnus Loop"], "snr", 7.0, 312.75, 30.72),
        ("NGC 2237", "Rosette Nebula", [], "en", 9.0, 97.98, 5.04),
        ("Mel 25", "Hyades", [], "oc", 0.5, 66.75, 15.87),
        ("NGC 5128", "Centaurus A", [], "gx", 6.8, 201.37, -43.02),
        ("NGC 253", "Sculptor Galaxy", [], "gx", 7.1, 11.89, -25.29),
        ("NGC 7000", "North America Nebula", [], "en", 4.0, 314.75, 44.33),
        ("C 1", "Kemble's Cascade", [], "ast", 5.0, 57.0, 63.0),
    ]
    for i, n, al, t, m, ra, dec in extras:
        dso.append({"id": i, "name": n, "aliases": al, "type": t, "mag": m, "ra": ra, "dec": dec})

    out = {"stars": stars, "cons": cons, "dso": dso}
    (DATA / "sky.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print(f"stars {len(stars)} (named {sum(1 for s in stars if s[4])}), constellations {len(cons)}, dso {len(dso)}; "
          f"{(DATA / 'sky.json').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
