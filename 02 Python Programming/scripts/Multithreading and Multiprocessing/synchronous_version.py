import requests
import time
from logging import basicConfig, info, INFO


DO_NOT_ABUSE_THE_SERVER = 40


def download_site(url: str, session):
    try:
        with session.get(url, timeout=10) as response:
            print(f"Read {len(response.content)} from {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    


def download_all_sites(sites):
    with requests.Session() as session:
        for url in sites:
            download_site(url, session)


if __name__ == "__main__":

    format = "%(levelname)s - %(message)s"
    basicConfig(
        filename="logging_result.log",
        level=INFO,
        format=format,
        filemode="a",
    )

    sites = [
        "https://www.jython.org",
        "https://devguide.python.org/internals/exploring/",
    ] * DO_NOT_ABUSE_THE_SERVER

    start_time = time.perf_counter()
    download_all_sites(sites)
    duration = time.perf_counter() - start_time

    info(f"Synchronous version downloaded {len(sites)} sites in {duration:.2f} seconds")

    print(f"Downloaded {len(sites)} in {duration:.2f} seconds")
