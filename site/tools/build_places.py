"""Pack a small gazetteer for set_location from GeoNames (CC-BY 4.0).

site/data/places.json:
  cities:  [[name, region, country, lat, lon, pop, tz], ...]   biggest first
  regions: {lowercase region/country name: index into cities of its capital or largest city}
  special: {lowercase name: [label, lat, lon]}  poles, observatories, deserts...
"""
import json
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
RAW = SITE.parent / "vendor" / "raw"     # cities15000.txt, countryInfo.txt, admin1CodesASCII.txt

ENGLISH_SPEAKING = {"US", "CA", "GB", "AU", "NZ", "IE", "IS", "NO"}

SPECIAL = {
    "the north pole": ["North Pole", 90.0, 0.0], "north pole": ["North Pole", 90.0, 0.0],
    "the south pole": ["South Pole", -90.0, 0.0], "south pole": ["South Pole", -90.0, 0.0],
    "the equator": ["the Equator", 0.0, -78.45], "equator": ["the Equator", 0.0, -78.45],
    "antarctica": ["McMurdo Station, Antarctica", -77.85, 166.67],
    "mauna kea": ["Mauna Kea, Hawaii", 19.82, -155.47],
    "the atacama desert": ["Atacama Desert, Chile", -24.5, -69.25], "atacama": ["Atacama Desert, Chile", -24.5, -69.25],
    "atacama desert": ["Atacama Desert, Chile", -24.5, -69.25],
    "patagonia": ["Patagonia, Argentina", -49.3, -72.9],
    "the arctic circle": ["Arctic Circle", 66.56, 25.85], "arctic circle": ["Arctic Circle", 66.56, 25.85],
    "greenwich": ["Royal Observatory Greenwich", 51.4769, -0.0005],
    "kitt peak": ["Kitt Peak, Arizona", 31.96, -111.6], "palomar": ["Palomar Observatory", 33.36, -116.86],
    "mount wilson": ["Mount Wilson Observatory", 34.22, -118.06], "la palma": ["La Palma, Canary Islands", 28.76, -17.89],
    "the sahara": ["Sahara Desert", 23.4, 12.0], "sahara": ["Sahara Desert", 23.4, 12.0],
    "tasmania": ["Hobart, Tasmania", -42.88, 147.33], "greenland": ["Nuuk, Greenland", 64.18, -51.72],
    "svalbard": ["Longyearbyen, Svalbard", 78.22, 15.65], "easter island": ["Easter Island", -27.11, -109.35],
    "uluru": ["Uluru, Australia", -25.34, 131.04], "the outback": ["Alice Springs, Australia", -23.7, 133.88],
}


def main():
    countries, capitals = {}, {}
    for line in (RAW / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        countries[f[0]] = f[4]
        capitals[f[0]] = f[5]
    admin1 = {}
    for line in (RAW / "admin1CodesASCII.txt").read_text(encoding="utf-8").splitlines():
        code, name, ascii_name, _ = line.split("\t")
        admin1[code] = ascii_name

    rows = []
    for line in (RAW / "cities15000.txt").read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        name, ascii_name, lat, lon, fcode, cc, a1, pop, tz = f[1], f[2], float(f[4]), float(f[5]), f[7], f[8], f[10], int(f[14] or 0), f[17]
        keep = pop >= 100_000 or fcode == "PPLC" or (cc in ENGLISH_SPEAKING and pop >= 25_000)
        if not keep:
            continue
        rows.append([ascii_name, admin1.get(f"{cc}.{a1}", ""), countries.get(cc, cc), round(lat, 3), round(lon, 3), pop, fcode, cc, tz])
    rows.sort(key=lambda r: -r[5])

    regions = {}
    for i, r in enumerate(rows):               # largest city stands in for its region
        if r[1]:
            regions.setdefault(r[1].lower(), i)
    for i, r in enumerate(rows):               # countries resolve to their capital when we have it
        if r[6] == "PPLC" or r[0] == capitals.get(r[7]):
            regions[r[2].lower()] = i
    for i, r in enumerate(rows):
        regions.setdefault(r[2].lower(), i)

    cities = [r[:6] + [r[8]] for r in rows]
    out = {"cities": cities, "regions": regions, "special": SPECIAL}
    path = SITE / "data" / "places.json"
    path.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print(f"{len(cities)} cities, {len(regions)} regions, {len(SPECIAL)} special; {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
