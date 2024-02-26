# Laboratory Work nr. 2 - Sorting Algorithms (QuickSort, MergeSort, HeapSort, GnomeSort)
# Student: Gusev Roman
# Group: FAF-222

import matplotlib.pyplot as plt
import time
import numpy as np
from prettytable import PrettyTable
from decimal import Decimal, getcontext
import sys
import random
import pandas


def median_of_three(arr, left, right):
    mid = (left + right) // 2
    if arr[left] > arr[mid]:
        arr[left], arr[mid] = arr[mid], arr[left]
    if arr[left] > arr[right]:
        arr[left], arr[right] = arr[right], arr[left]
    if arr[mid] > arr[right]:
        arr[mid], arr[right] = arr[right], arr[mid]
    return mid


def partition(arr, left, right):
    pivot_index = median_of_three(arr, left, right)
    pivot = arr[pivot_index]
    arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
    i = left - 1
    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1


def quicksort(arr):
    def _quicksort(arr, left, right):
        if left < right:
            pivot_index = partition(arr, left, right)
            _quicksort(arr, left, pivot_index - 1)
            _quicksort(arr, pivot_index + 1, right)

    _quicksort(arr, 0, len(arr) - 1)


def gnome_sort(arr, n):
    index = 0
    while index < n:
        if index == 0:
            index = index + 1
        if arr[index] >= arr[index - 1]:
            index = index + 1
        else:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            index = index - 1

    return arr


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        lefthalf = arr[:mid].copy()
        righthalf = arr[mid:].copy()

        merge_sort(lefthalf)
        merge_sort(righthalf)

        i = 0
        j = 0
        k = 0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                arr[k] = lefthalf[i]
                i = i + 1
            else:
                arr[k] = righthalf[j]
                j = j + 1
            k = k + 1

        while i < len(lefthalf):
            arr[k] = lefthalf[i]
            i = i + 1
            k = k + 1

        while j < len(righthalf):
            arr[k] = righthalf[j]
            j = j + 1
            k = k + 1


def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    # See if left child of root exists and is
    # greater than root
    if l < n and arr[i] < arr[l]:
        largest = l

    # See if right child of root exists and is
    # greater than root
    if r < n and arr[largest] < arr[r]:
        largest = r

    # Change root, if needed
    if largest != i:
        (arr[i], arr[largest]) = (arr[largest], arr[i])  # swap

        # Heapify the root.
        heapify(arr, n, largest)


def heap_sort(arr):
    n = len(arr)

    # Build a maxheap.
    # Since last parent will be at (n//2) we can start at that location.

    for i in range(n // 2, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements

    for i in range(n - 1, 0, -1):
        (arr[i], arr[0]) = (arr[0], arr[i])  # swap
        heapify(arr, i, 0)


def clear_lists():
    pretty_table.clear()
    time_values.clear()
    time_values_short.clear()


def plot_graph(sort_name, size_lst, time_values_lst):
    font = {'size': 15}
    plt.rc('font', **font)
    plt.plot(size_lst,
             time_values_lst,
             marker="o")
    plt.title(f"Growth of Time Complexity in {sort_name}")
    plt.xlabel("Size")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.show()


def print_single_algorithm(sort_name, time_values_dict):
    pretty_table.add_column("Algorithm", [sort_name])
    for (name, elapsed_time) in time_values_dict.items():
        pretty_table.add_column(name, ["{:e}".format(elapsed_time)])


if __name__ == '__main__':
    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    small_sizes = [100, 500, 1000, 5000, 10000]
    # sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 2000000]

    randomized_array_100 = np.array([random.randint(-5000001, 5000001) for _ in range(100)])
    randomized_array_500 = np.array([random.randint(-5000001, 5000001) for _ in range(500)])
    randomized_array_1k = np.array([random.randint(-5000001, 5000001) for _ in range(1000)])
    randomized_array_5k = np.array([random.randint(-5000001, 5000001) for _ in range(5000)])
    randomized_array_10k = np.array([random.randint(-5000001, 5000001) for _ in range(10000)])
    randomized_array_50k = np.array([random.randint(-5000001, 5000001) for _ in range(50000)])
    randomized_array_100k = np.array([random.randint(-5000001, 5000001) for _ in range(100000)])
    randomized_array_500k = np.array([random.randint(-5000001, 5000001) for _ in range(500000)])
    randomized_array_1m = np.array([random.randint(-5000001, 5000001) for _ in range(1000000)])
    # randomized_array_2m =   np.array([random.randint(-5000001, 5000001) for _ in range(2000000)])

    sorted_ascending_array = np.sort(randomized_array_1m)
    sorted_descending_array = np.sort(randomized_array_1m)[::-1]

    arrays_short = {
        "100": randomized_array_100,
        "500": randomized_array_500,
        "1000": randomized_array_1k,
        "5000": randomized_array_5k,
        "10000": randomized_array_10k
    }

    arrays_long = {
        "100": randomized_array_100,
        "500": randomized_array_500,
        "1000": randomized_array_1k,
        "5000": randomized_array_5k,
        "10000": randomized_array_10k,
        "50000": randomized_array_50k,
        "100000": randomized_array_100k,
        "500000": randomized_array_500k,
        "1000000": randomized_array_1m,
        # "2000000": randomized_array_2m,
        "Sorted Ascending 1M": sorted_ascending_array,
        "Sorted Descending 1M": sorted_descending_array
    }

    all_values_short = {
        "Size": small_sizes,
        "QuickSort": [],
        "GnomeSort": [],
        "MergeSort": [],
        "HeapSort": []
    }

    all_values = {
        "Size": sizes + ["Sorted Ascending 1M", "Sorted Descending 1M"],
        "QuickSort": [],
        "MergeSort": [],
        "HeapSort": []
    }
    pretty_table_all_long = PrettyTable()
    pretty_table_all_short = PrettyTable()

    pretty_table = PrettyTable()

    time_values = {}
    time_values_short = {}

    for (name, array) in arrays_long.items():
        new_array = array.copy()

        start_time = time.perf_counter()
        quicksort(new_array)
        elapsed_time = time.perf_counter()
        elapsed_time = elapsed_time - start_time

        time_values[name] = elapsed_time

        if len(new_array) == 100:
            print(new_array)

        if len(new_array) <= 10000:
            time_values_short[name] = elapsed_time

    print_single_algorithm("QuickSort", time_values)
    print(pretty_table)

    pretty_table.clear()
    print_single_algorithm("QuickSort", time_values_short)
    print(pretty_table)

    values = list(time_values.values())
    temp = list(time_values.keys()).index("Sorted Ascending 1M")
    res = values[:temp]
    plot_graph("QuickSort", sizes, res)

    values = list(time_values_short.values())
    plot_graph("QuickSort", small_sizes, values)

    all_values["QuickSort"] = list(time_values.values())
    all_values_short["QuickSort"] = list(time_values_short.values())

    clear_lists()

    # GnomeSort
    for (name, array) in arrays_short.items():
        new_array = array.copy()

        start_time = time.perf_counter()
        gnome_sort(new_array, len(new_array))
        elapsed_time = time.perf_counter()
        elapsed_time = elapsed_time - start_time

        time_values_short[name] = elapsed_time

        if len(new_array) == 100:
            print(new_array)

    print_single_algorithm("GnomeSort", time_values_short)
    print(pretty_table)

    values = time_values_short.values()
    plot_graph("GnomeSort", small_sizes, values)

    all_values_short["GnomeSort"] = list(time_values_short.values())

    clear_lists()

    # MergeSort
    for (name, array) in arrays_long.items():
        new_array = array.copy()

        start_time = time.perf_counter()
        merge_sort(new_array)
        elapsed_time = time.perf_counter()
        elapsed_time = elapsed_time - start_time

        time_values[name] = elapsed_time

        if len(new_array) == 100:
            print(new_array)

        if len(new_array) <= 10000:
            time_values_short[name] = elapsed_time

    print_single_algorithm("MergeSort", time_values)
    print(pretty_table)

    pretty_table.clear()
    print_single_algorithm("MergeSort", time_values_short)
    print(pretty_table)

    values = list(time_values.values())
    temp = list(time_values.keys()).index("Sorted Ascending 1M")
    res = values[:temp]
    plot_graph("MergeSort", sizes, res)

    values = list(time_values_short.values())
    plot_graph("MergeSort", small_sizes, values)

    all_values["MergeSort"] = list(time_values.values())
    all_values_short["MergeSort"] = list(time_values_short.values())

    clear_lists()

    # HeapSort
    for (name, array) in arrays_long.items():
        new_array = array.copy()

        start_time = time.perf_counter()
        heap_sort(new_array)
        elapsed_time = time.perf_counter()
        elapsed_time = elapsed_time - start_time

        time_values[name] = elapsed_time

        if len(new_array) == 100:
            print(new_array)

        if len(new_array) <= 10000:
            time_values_short[name] = elapsed_time

    print_single_algorithm("HeapSort", time_values)
    print(pretty_table)

    pretty_table.clear()
    print_single_algorithm("HeapSort", time_values_short)
    print(pretty_table)

    values = list(time_values.values())
    temp = list(time_values.keys()).index("Sorted Ascending 1M")
    res = values[:temp]
    plot_graph("HeapSort", sizes, res)

    values = list(time_values_short.values())
    plot_graph("HeapSort", small_sizes, values)

    all_values["HeapSort"] = list(time_values.values())
    all_values_short["HeapSort"] = list(time_values_short.values())

    clear_lists()

    # Output Comparison Tables
    for (key, value) in all_values_short.items():
        pretty_table_all_short.add_column(key, value)
    print(pretty_table_all_short)

    for (key, value) in all_values.items():
        pretty_table_all_long.add_column(key, value)
    print(pretty_table_all_long)

    # Output Graph Comparison
    df = pandas.DataFrame(all_values_short)

    for sort in ["QuickSort", "HeapSort", "MergeSort", "GnomeSort"]:
        plt.plot(df["Size"], df[sort], label=sort)
    plt.xlabel('Size')
    plt.ylabel('Time (seconds)')
    plt.title('Comparison of Sorting Algorithms on Random Data (Short)')
    plt.legend()
    plt.grid(True)
    plt.show()

    dic = {"Size": sizes}
    for (key, value) in all_values.items():
        if key == "Size":
            temp = value.index("Sorted Ascending 1M")
            res = value[:temp]
            dic[key] = res
        else:
            temp = len(value) - 2
            res = value[:temp]
            dic[key] = res

    df = pandas.DataFrame(dic)
    print(df)
    for sort in ["QuickSort", "HeapSort", "MergeSort"]:
        plt.plot(df["Size"], df[sort], label=sort)
    plt.xlabel('Size')
    plt.ylabel('Time (seconds)')
    plt.title('Comparison of Sorting Algorithms on Random Data (Short)')
    plt.legend()
    plt.grid(True)
    plt.show()
