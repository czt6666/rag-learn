import time
import asyncio


# 1. asyncio: 单线程+事件循环 - asyncio.run(main())
# 2. threading: 单进程+多线程 - threading.Thread()
# 3. multiprocessing: 多进程 - Process()
# 4. ThreadPoolExecutor: 线程池
# 5. ProcessPoolExecutor: 进程池

async def task(name):
    print(f"{name} start")
    await asyncio.sleep(1)
    print(f"{name} end")


async def main():
    await asyncio.gather(
        task("A"),
        task("B"),
        task("C"),
    )


# asyncio.run(main())


def task(name):
    print(f"{name} start")
    time.sleep(1)
    print(f"{name} end")


# task("A")
# task("B")
# task("C")

import threading

# def work(name):
#     print(f"working {name}")
#
#
# t1 = threading.Thread(target=work, args=("A",))
# t2 = threading.Thread(target=work, args=("B",))
#
# t1.start()
# t2.start()
#
# t1.join()
# t2.join()

from multiprocessing import Process


def work(name):
    print(f"working {name}")


# p1 = Process(target=work, args=("A",))
# p2 = Process(target=work, args=("B",))
#
# p1.start()
# p2.start()
#
# p1.join()
# p2.join()

threads = []
for i in range(5):
    t = threading.Thread(target=work, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

processes = []
for i in range(5):
    p = Process(target=work, args=(i,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as pool:
    pool.map(work, range(10))

from concurrent.futures import ProcessPoolExecutor
