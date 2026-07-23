#!/usr/bin/env python3
"""
Pulls historical closing NHL totals (over/under line + American odds) from
The Odds API for every ELIGIBLE game in data_walkforward_2025-26.csv, and
writes the result to a CSV.

This must be run on YOUR machine with YOUR API key — this session's network
access is blocked from reaching api.the-odds-api.com, so this script cannot
be executed here. Only stdlib is used (urllib), so no pip install needed.

Usage:
    export ODDS_API_KEY=your_key_here
    python3 fetch_historical_odds.py --limit 5          # pilot run first
    python3 fetch_historical_odds.py                    # full run
    python3 fetch_historical_odds.py --resume           # skip games already in output CSV

Notes on cost:
    Historical requests typically cost far more credits than live ones
    (commonly ~10x per market/region on The Odds API's pricing). This script
    prints x-requests-remaining after every call and stops automatically if
    remaining credits drop below --min-credits, so you don't blow through
    your quota by accident. Check the-odds-api.com's pricing/plan page
    yourself to confirm your plan even includes the historical endpoint —
    if it doesn't, every call below will fail with a 401/403 and the script
    will tell you clearly rather than retrying blindly.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:
    print("Python 3.9+ required (for zoneinfo).", file=sys.stderr)
    sys.exit(1)

API_BASE = "https://api.the-odds-api.com/v4/historical/sports/icehockey_nhl/odds"
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Prefer a specific book if present, else fall back to the first one returned.
PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "pinnacle"]


def load_eligible_games(path):
    games = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Eligible"] == "True":
                games.append(row)
    return games


def load_game_times(path):
    """Maps (Date, Away, Home) -> Time string from the original schedule CSV,
    since the walk-forward CSV doesn't carry game time."""
    times = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0]:
                continue
            date, time_, away, ag, home, hg = row[0], row[1], row[2], row[3], row[4], row[5]
            times[(date, away, home)] = time_
    return times


def load_already_fetched(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["Date"], row["Away"], row["Home"]))
    return done


def to_commence_utc(date_str, time_str):
    # date_str like "2025-10-07", time_str like "5:00 PM" (US Eastern, per Hockey-Reference convention)
    naive = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %I:%M %p")
    eastern = naive.replace(tzinfo=EASTERN)
    return eastern.astimezone(UTC)


def api_get(url, api_key_note_only=False):
    req = urllib.request.Request(url, headers={"User-Agent": "gino-gauge-backtest/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            remaining = resp.headers.get("x-requests-remaining")
            used = resp.headers.get("x-requests-used")
            return json.loads(body), remaining, used, resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": body}, e.headers.get("x-requests-remaining"), e.headers.get("x-requests-used"), e.code


def extract_totals(event, away, home):
    for bm in event.get("bookmakers", []):
        pass
    # try preferred books first
    books = {bm["key"]: bm for bm in event.get("bookmakers", [])}
    ordered_keys = [k for k in PREFERRED_BOOKS if k in books] + [k for k in books if k not in PREFERRED_BOOKS]
    for key in ordered_keys:
        bm = books[key]
        for market in bm.get("markets", []):
            if market["key"] == "totals":
                over = next((o for o in market["outcomes"] if o["name"] == "Over"), None)
                under = next((o for o in market["outcomes"] if o["name"] == "Under"), None)
                if over and under:
                    return {
                        "bookmaker": key,
                        "point": over.get("point"),
                        "over_price": over.get("price"),
                        "under_price": under.get("price"),
                    }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_walkforward_2025-26.csv")
    ap.add_argument("--games-csv", default="data_games_2025-26.csv",
                     help="Original schedule CSV, used to look up each game's actual start time")
    ap.add_argument("--output", default="data_odds_2025-26.csv")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N games (pilot run)")
    ap.add_argument("--resume", action="store_true", help="Skip games already present in --output")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between API calls")
    ap.add_argument("--min-credits", type=int, default=5, help="Abort if remaining credits drop below this")
    ap.add_argument("--regions", default="us")
    args = ap.parse_args()

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("Set ODDS_API_KEY as an environment variable first (do not hardcode it in this file).",
              file=sys.stderr)
        sys.exit(1)

    games = load_eligible_games(args.input)
    game_times = load_game_times(args.games_csv)
    already = load_already_fetched(args.output) if args.resume else set()

    write_header = not (args.resume and os.path.exists(args.output))
    out_f = open(args.output, "a" if args.resume else "w", newline="")
    writer = csv.writer(out_f)
    if write_header:
        writer.writerow(["Date", "Away", "Home", "Total_Line", "Over_Odds", "Under_Odds", "Bookmaker", "Snapshot_UTC"])
        out_f.flush()

    processed = 0
    for g in games:
        key = (g["Date"], g["Away"], g["Home"])
        if key in already:
            continue
        if args.limit is not None and processed >= args.limit:
            break

        time_str = game_times.get((g["Date"], g["Away"], g["Home"]))
        if not time_str:
            print(f"[{g['Date']}] {g['Away']} @ {g['Home']} -> no matching row in {args.games_csv}, skipping",
                  file=sys.stderr)
            processed += 1
            continue
        commence_utc = to_commence_utc(g["Date"], time_str)
        query_ts = commence_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        url = (f"{API_BASE}?apiKey={api_key}&regions={args.regions}"
               f"&markets=totals&oddsFormat=american&date={query_ts}")

        data, remaining, used, status = api_get(url)

        if status != 200:
            print(f"[{g['Date']}] {g['Away']} @ {g['Home']} -> HTTP {status}: {data.get('error', data)}",
                  file=sys.stderr)
            if status in (401, 403):
                print("Your API key/plan likely doesn't include historical access. "
                      "Check the-odds-api.com's pricing page. Stopping.", file=sys.stderr)
                break
            processed += 1
            time.sleep(args.sleep)
            continue

        print(f"[{g['Date']}] {g['Away']} @ {g['Home']} -> requests remaining: {remaining} (used: {used})")

        events = data.get("data", []) if isinstance(data, dict) else []
        match = None
        for ev in events:
            if ev.get("away_team") == g["Away"] and ev.get("home_team") == g["Home"]:
                match = ev
                break

        if match:
            totals = extract_totals(match, g["Away"], g["Home"])
            if totals:
                writer.writerow([g["Date"], g["Away"], g["Home"], totals["point"],
                                  totals["over_price"], totals["under_price"],
                                  totals["bookmaker"], data.get("timestamp", "")])
            else:
                print(f"  no totals market found for this game at this snapshot", file=sys.stderr)
        else:
            print(f"  no matching event found in snapshot ({len(events)} events returned)", file=sys.stderr)

        out_f.flush()
        processed += 1

        if remaining is not None and remaining.isdigit() and int(remaining) < args.min_credits:
            print(f"Remaining credits ({remaining}) below --min-credits ({args.min_credits}). Stopping.",
                  file=sys.stderr)
            break

        time.sleep(args.sleep)

    out_f.close()
    print(f"\nDone. Processed {processed} games this run. Output: {args.output}")


if __name__ == "__main__":
    main()
