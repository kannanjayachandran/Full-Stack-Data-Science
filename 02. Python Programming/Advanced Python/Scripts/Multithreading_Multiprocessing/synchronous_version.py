import requests
import time
from logging import basicConfig, info, INFO


DO_NOT_ABUSE_THE_SERVER = 40


def download_site(url, session):
    with session.get(url) as response:
        print(f"Read {len(response.content)} from {url}")


def download_all_sites(sites):
    with requests.Session() as session:
        for url in sites:
            download_site(url, session)


if __name__ == "__main__":

    # logging configuration
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

    start_time = time.time()
    download_all_sites(sites)
    duration = time.time() - start_time

    info(f"Synchronous_Version downloaded {len(sites)} sites in {duration:.2f} seconds")

    print(f"Downloaded {len(sites)} in {duration:.2f} seconds")
