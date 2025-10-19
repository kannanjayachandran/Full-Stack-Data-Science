# asyncio version
import asyncio
import time
import aiohttp
from logging import basicConfig, info, INFO


DO_NOT_ABUSE_THE_SERVER = 40


async def download_site(session, url):
    try:
        async with session.get(url) as response:
            content = await response.read()
            print(f"Read {len(content)} from {url}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")

async def download_all_sites(sites):
    async with aiohttp.ClientSession() as session:
        task = [asyncio.create_task(download_site(session, url)) for url in sites]
        await asyncio.gather(*task, return_exceptions=True)


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
    asyncio.run(download_all_sites(sites))
    duration = time.perf_counter() - start_time

    info(f"Asyncio version downloaded {len(sites)} sites in {duration:.2f} seconds")

    print(f"Downloaded {len(sites)} in {duration:.2f} seconds")
