# Laboratory Work nr. 1 - Fibonacci Numbers Finding Algorithms
# Student: Gusev Roman
# Group: FAF-222

import matplotlib.pyplot as plt
import time
from prettytable import PrettyTable
from decimal import Decimal, getcontext
import sys


def fibonacci_recursive(nth_term):
    if nth_term <= 1:
        return nth_term
    return fibonacci_recursive(nth_term - 1) + fibonacci_recursive(nth_term - 2)


def fibonacci_iterative(nth_term):
    f_0 = 0
    f_1 = 1
    if nth_term == 0:
        return f_0
    elif nth_term == 1:
        return f_1
    else:
        for _ in range(2, nth_term + 1):
            f_next = f_0 + f_1
            f_0 = f_1
            f_1 = f_next
        return f_1


sys.setrecursionlimit(10 ** 6)
def fibonacci_dynamic_list(nth_term, memo={}):
    if nth_term in memo:
        return memo[nth_term]
    if nth_term <= 2:
        return 1

    memo[nth_term] = fibonacci_dynamic_list(nth_term - 1, memo) + fibonacci_dynamic_list(nth_term - 2, memo)
    return memo[nth_term]


def fibonacci_matrix(nth_term):
    F = [[1, 1],
         [1, 0]]
    if nth_term == 0:
        return 0
    power(F, nth_term - 1)

    return F[0][0]


def multiply(F, M):
    x = (F[0][0] * M[0][0] +
         F[0][1] * M[1][0])
    y = (F[0][0] * M[0][1] +
         F[0][1] * M[1][1])
    z = (F[1][0] * M[0][0] +
         F[1][1] * M[1][0])
    w = (F[1][0] * M[0][1] +
         F[1][1] * M[1][1])

    F[0][0] = x
    F[0][1] = y
    F[1][0] = z
    F[1][1] = w


def power(F, n):
    if n == 0 or n == 1:
        return
    M = [[1, 1],
         [1, 0]]

    power(F, n // 2)
    multiply(F, F)

    if n % 2 != 0:
        multiply(F, M)


def fibonacci_binet(nth_term):
    # getcontext().prec = 1000
    square_root = Decimal(5).sqrt()
    phi = (1 + square_root) / 2
    psi = (1 - square_root) / 2
    val = (phi ** nth_term - psi ** nth_term) / square_root
    return val


def fibonacci_fast_doubling_main(nth_term):
    return fibonacci_fast_doubling(nth_term)[0]


def fibonacci_fast_doubling(nth_term):
    if nth_term == 0:
        return 0, 1
    else:
        a, b = fibonacci_fast_doubling(nth_term // 2)
        c = a * (b * 2 - a)
        d = a * a + b * b
        if nth_term % 2 == 0:
            return c, d
        else:
            return d, c + d


def print_long():
    pretty_table.add_column("nth Term", terms_list_long)
    pretty_table.add_column("Time(s)", time_values)
    print(pretty_table)


def print_short():
    pretty_table.add_column("nth Term", terms_list_short)
    pretty_table.add_column("Time(s)", time_values)
    pretty_table.add_column("Value", fibonacci_values)
    print(pretty_table)


def clear_lists():
    pretty_table.clear()
    fibonacci_values.clear()
    time_values.clear()


def print_short_comparison(type_list):
    comparison_table.add_column("nth Term", terms_list_short if type_list == "short" else terms_list_long)
    for key, word in all_values.items():
        print(key, word)
        comparison_table.add_column(key, word)
    print(comparison_table)


def plot_methods_values(type_list, function_name):
    font = {'size': 15}
    plt.rc('font', **font)
    plt.plot(terms_list_short if type_list == "short" else terms_list_long,
             time_values,
             marker="o")
    plt.title(f"Growth of Time Complexity in {function_name} Approach for Fibonacci Series")
    plt.xlabel("n-th term")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    comparison_table = PrettyTable()

    pretty_table = PrettyTable()
    time_values = []
    fibonacci_values = []

    all_values = {}

    terms_list_short = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45]
    terms_list_long = [500, 1000, 1585, 2512, 4000, 6310, 10000, 15849, 25000, 50000, 100000, 150000, 200000, 250000]

    # RECURSIVE APPROACH
    # print("APPROACH: RECURSIVE")
    #
    # for term in terms_list_short:
    #     start_time = time.perf_counter()
    #     value = fibonacci_recursive(term)
    #     elapsed_time = time.perf_counter()
    #     elapsed_time = elapsed_time - start_time
    #     time_values.append(elapsed_time)
    #     fibonacci_values.append(value)
    #
    # all_values["Recursive"] = time_values.copy()
    # print(time_values)
    # plot_methods_values("short", "Recursive")
    #
    # print_short()
    #
    # # ITERATIVE APPROACH
    # print("APPROACH: ITERATIVE")
    # clear_lists()
    #
    # for term in terms_list_short:
    #     start_time = time.perf_counter()
    #     value = fibonacci_iterative(term)
    #     elapsed_time = time.perf_counter()
    #     elapsed_time = elapsed_time - start_time
    #     time_values.append(elapsed_time)
    #     fibonacci_values.append(value)
    #
    # all_values["Iterative"] = time_values.copy()
    # print(time_values)
    # plot_methods_values("short", "Iterative")
    #
    # print_short()

    # DYNAMIC PROGRAMMING APPROACH
    print("APPROACH: DYNAMIC PROGRAMMING")
    clear_lists()

    for term in terms_list_short:
        start_time = time.perf_counter()
        value = fibonacci_dynamic_list(term)
        elapsed_time = time.perf_counter()
        elapsed_time = elapsed_time - start_time
        time_values.append(elapsed_time)
        fibonacci_values.append(value)

    all_values["Dynamic Programming"] = time_values.copy()
    plot_methods_values("short", "Dynamic Programming")

    print_short()

    # Nth POWER OF MATRIX APPROACH
    # print("APPROACH: Nth POWER OF MATRIX")
    # clear_lists()
    # for term in terms_list_short:
    #     start_time = time.perf_counter()
    #     value = fibonacci_matrix(term)
    #     elapsed_time = time.perf_counter()
    #     elapsed_time = elapsed_time - start_time
    #     time_values.append(elapsed_time)
    #     fibonacci_values.append(value)
    #
    # all_values["Matrix"] = time_values.copy()
    # plot_methods_values("short", " Nth Power of Matrix Approach")
    #
    # print_short()
    #
    # # BINET FORMULA APPROACH
    # print("APPROACH: BINET'S FORMULA")
    # clear_lists()
    #
    # for term in terms_list_short:
    #     start_time = time.perf_counter()
    #     value = fibonacci_binet(term)
    #     elapsed_time = time.perf_counter()
    #     elapsed_time = elapsed_time - start_time
    #     time_values.append(elapsed_time)
    #     fibonacci_values.append(value)
    #
    # all_values["Binet's Formula"] = time_values.copy()
    # plot_methods_values("short", "Binet's Formula")
    #
    # print_short()
    #
    # # FAST DOUBLING APPROACH
    # print("APPROACH: FAST DOUBLING")
    # clear_lists()
    #
    # for term in terms_list_short:
    #     start_time = time.perf_counter()
    #     value = fibonacci_fast_doubling_main(term)
    #     elapsed_time = time.perf_counter()
    #     elapsed_time = elapsed_time - start_time
    #     time_values.append(elapsed_time)
    #     fibonacci_values.append(value)
    #
    # all_values["Fast Doubling"] = time_values.copy()
    # plot_methods_values("short", "Fast Doubling")
    #
    # print_short()
    #
    # print_short_comparison("short")
    #
    # # ITERATIVE APPROACH
    # print("APPROACH: ITERATIVE")
    # clear_lists()
    # all_values.clear()
    # comparison_table.clear()
    #
    # for term in terms_list_long:
    #     start_time = time.perf_counter()
    #     value = fibonacci_iterative(term)
    #     elapsed_time = time.perf_counter()
    #     elapsed_time = elapsed_time - start_time
    #     time_values.append(elapsed_time)
    #     fibonacci_values.append(value)
    #
    # all_values["Iterative"] = time_values.copy()
    # plot_methods_values("long", "Iterative")
    #
    # print_long()
    #
    # # DYNAMIC PROGRAMMING APPROACH
    # print("APPROACH: DYNAMIC PROGRAMMING")
    # clear_lists()
    #
    # for term in terms_list_long:
    #     start_time = time.perf_counter()
    #     value = fibonacci_dynamic_list(term)
    #     elapsed_time = time.perf_counter()
    #     elapsed_time = elapsed_time - start_time
    #     time_values.append(elapsed_time)
    #     fibonacci_values.append(value)
    #
    # all_values["Dynamic Programming"] = time_values.copy()
    # plot_methods_values("long", "Dynamic Programming")
    #
    # print_long()
    #
    # # Nth POWER OF MATRIX APPROACH
    # print("APPROACH: Nth POWER OF MATRIX")
    # clear_lists()
    # for term in terms_list_long:
    #     start_time = time.perf_counter()
    #     value = fibonacci_matrix(term)
    #     elapsed_time = time.perf_counter()
    #     elapsed_time = elapsed_time - start_time
    #     time_values.append(elapsed_time)
    #     fibonacci_values.append(value)
    #
    # all_values["Matrix"] = time_values.copy()
    # plot_methods_values("long", " Nth Power of Matrix Approach")
    #
    # print_long()

    # BINET FORMULA APPROACH
    print("APPROACH: BINET'S FORMULA")
    clear_lists()

    for term in terms_list_long:
        start_time = time.perf_counter()
        value = fibonacci_binet(term)
        elapsed_time = time.perf_counter()
        elapsed_time = elapsed_time - start_time
        time_values.append(elapsed_time)
        fibonacci_values.append(value)
    all_values["Binet's Formula"] = time_values.copy()

    plot_methods_values("long", "Binet's Formula")

    print_long()

    # FAST DOUBLING APPROACH
    # print("APPROACH: FAST DOUBLING")
    # clear_lists()
    #
    # for term in terms_list_long:
    #     start_time = time.perf_counter()
    #     value = fibonacci_fast_doubling_main(term)
    #     elapsed_time = time.perf_counter()
    #     elapsed_time = elapsed_time - start_time
    #     time_values.append(elapsed_time)
    #     fibonacci_values.append(value)
    #
    # all_values["Fast Doubling"] = time_values.copy()
    # plot_methods_values("long", "Fast Doubling")
    #
    # print_long()
    #
    # print_short_comparison("long")
