"""How many cars are actually on the road in our SF chunk, by time of day.

We estimate the INSTANTANEOUS number of moving vehicles, then distribute them
evenly across the network by lane-length (cars per lane-km). The method:

    N(t) = daily_VMT * hour_fraction(t) / v_avg(t)        [citywide moving cars]
    density(t) = N(t) / city_lane_km                      [cars per lane-km]
    chunk_cars(t) = density(t) * chunk_lane_km            [our area, evenly spread]

Citywide inputs are defensible public-data estimates (stated as assumptions, with
ranges), not exact official figures — the point is a grounded number, not 300.

Sources / assumptions:
  * SF within-city daily VMT ~ 12 million veh-mi/day (MTC Vital Signs / SFMTA
    range ~11-13M). https://www.vitalsigns.mtc.ca.gov/  (SF County VMT)
  * SF ~1,200 miles of streets (SFMTA) -> ~1,930 km centerline; avg ~2 lanes
    + freeways -> ~4,000 lane-km citywide.
  * Hourly traffic fractions (K-factor) and travel speeds are typical urban
    weekday values (Caltrans/SFCTA congestion ranges).
"""
from __future__ import annotations

DAILY_VMT_SF = 12_000_000.0     # veh-miles / day, within SF (assumption, ~11-13M)
CITY_LANE_KM = 4_000.0          # citywide lane-km (assumption)

# time-of-day: hour -> (fraction of daily traffic in that hour, avg speed mph)
PROFILE = {
    "3am (deep night)":   (0.010, 28.0),
    "6am (early)":        (0.035, 26.0),
    "8am (AM peak)":      (0.085, 14.0),
    "noon (midday)":      (0.055, 22.0),
    "3pm (afternoon)":    (0.065, 19.0),
    "5pm (PM peak)":      (0.090, 13.0),
    "8pm (evening)":      (0.045, 23.0),
    "midnight":           (0.015, 27.0),
}


def city_moving_cars(frac: float, v_avg_mph: float) -> float:
    return DAILY_VMT_SF * frac / v_avg_mph


def density_per_lane_km(frac: float, v_avg_mph: float) -> float:
    return city_moving_cars(frac, v_avg_mph) / CITY_LANE_KM


def chunk_cars(chunk_lane_km: float, label: str) -> float:
    frac, v = PROFILE[label]
    return density_per_lane_km(frac, v) * chunk_lane_km


def report(chunk_lane_km: float):
    print(f"citywide assumptions: {DAILY_VMT_SF/1e6:.0f}M VMT/day, "
          f"{CITY_LANE_KM:.0f} lane-km")
    print(f"chunk: {chunk_lane_km:.0f} lane-km\n")
    print(f"{'time of day':<20}{'city cars':>12}{'cars/lane-km':>14}"
          f"{'CHUNK cars':>12}{'1 car / m':>11}")
    print("-" * 69)
    for label, (frac, v) in PROFILE.items():
        cc = city_moving_cars(frac, v)
        dens = cc / CITY_LANE_KM
        chunk = dens * chunk_lane_km
        headway = 1000.0 / dens if dens > 0 else float("inf")
        print(f"{label:<20}{cc:>12,.0f}{dens:>14.1f}{chunk:>12,.0f}"
              f"{headway:>10.0f}m")


if __name__ == "__main__":
    from .map_loader import load_sf_graph, to_road_network
    BIG = (-122.4300, 37.7250, -122.3800, 37.8050)
    net = to_road_network(load_sf_graph(bbox=BIG, cache_name="sf_huge_drive.graphml"))
    lane_km = (net.edge_length * net.edge_lanes).sum() / 1000 / 2
    report(lane_km)
