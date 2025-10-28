# threading version
import concurrent.futures
import requests
import threading
import time
from logging import basicConfig, info, INFO


DO_NOT_ABUSE_THE_SERVER = 40
thread_local = threading.local()


def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def download_site(url):
    session = get_session()
    try:
        with session.get(url, timeout=10) as response:
            print(f"Thread {threading.current_thread().name} read {len(response.content)} from {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def download_all_sites(sites):
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(download_site, sites)


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

    info(
        f"Multi-threading with concurrent.futures downloaded {len(sites)} sites in {duration:.2f} seconds"
    )

    print(f"Downloaded {len(sites)} in {duration} seconds")
