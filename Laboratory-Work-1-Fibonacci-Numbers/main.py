# Laboratory Work nr. 1 - Fibonacci Numbers Finding Algorithms
# Student: Gusev Roman
# Group: FAF-222

import matplotlib.pyplot as plt
import time
from prettytable import PrettyTable


def fibonacci_recursive(nth_term):
    if nth_term < 0:
        return "Invalid input"
    if nth_term <= 1:
        return nth_term
    return fibonacci_recursive(nth_term - 1) + fibonacci_recursive(nth_term - 2)


def fibonacci_iterative(nth_term):
    f_0 = 0
    f_1 = 1
    if nth_term < 0:
        return "Invalid input"
    elif nth_term == 0:
        return f_0
    elif nth_term == 1:
        return f_1
    else:
        for _ in range(2, nth_term + 1):
            f_next = f_0 + f_1
            f_0 = f_1
            f_1 = f_next
        return f_1


def fibonacci_dynamic(nth_term):
    lst = [0, 1]

    for current_term in range(2, nth_term + 1):
        lst.append(lst[current_term - 1] + lst[current_term - 2])
    return lst[nth_term]


if __name__ == '__main__':
    pretty_table = PrettyTable()
    time_values = []
    fibonacci_values = []

    terms_list_short = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45]
    terms_list_long = [500, 1000, 1585, 2512, 4000, 6310, 10000, 15849, 25000, 50000, 100000]

    # RECURSIVE APPROACH

    # for term in terms_list_short:
    #     start_time = time.perf_counter()
    #     value = fibonacci_recursive(term)
    #     elapsed_time = time.perf_counter()
    #     elapsed_time = round(elapsed_time - start_time, 6)
    #     time_values.append(elapsed_time)
    #     fibonacci_values.append(value)
    #
    # plt.plot(terms_list_short,
    #          time_values,
    #          marker="o")
    # plt.title("Growth of Time Complexity in Recursive Fibonacci Series")
    # plt.xlabel("n-th term")
    # plt.ylabel("Time (s)")
    # plt.grid(True)
    # plt.show()
    #
    # pretty_table.field_names = ["n"] + terms_list_short
    # pretty_table.add_row(["Time(s)"] + time_values)
    # print(pretty_table)

    # ITERATIVE APPROACH
    # pretty_table.clear()
    # fibonacci_values.clear()
    # time_values.clear()
    # for term in terms_list_short:
    #     start_time = time.perf_counter()
    #     value = fibonacci_iterative(term)
    #     elapsed_time = time.perf_counter()
    #     elapsed_time = elapsed_time - start_time
    #     time_values.append(elapsed_time)
    #     fibonacci_values.append(value)
    #
    # plt.plot(terms_list_short,
    #          time_values,
    #          marker="o")
    # plt.title("Growth of Time Complexity in Iterative Algorithm for Fibonacci Series")
    # plt.xlabel("n-th term")
    # plt.ylabel("Time (s)")
    # plt.grid(True)
    # for i in range(len(terms_list_short)):
    #     plt.text(terms_list_short[i],
    #              time_values[i] + 0.00000003,
    #              str(f"n={terms_list_short[i]} | ts={round(time_values[i], 12)}"),
    #              ha='center',
    #              bbox=dict(
    #                  facecolor='white',
    #                  edgecolor='black',
    #                  boxstyle='round,pad=0.5')
    #              )
    # plt.show()
    #
    # pretty_table.field_names = ["n"] + terms_list_short
    # pretty_table.add_row(["Time(s)"] + time_values)
    # print(pretty_table)

    # DYNAMIC PROGRAMMING APPROACH
    pretty_table.clear()
    fibonacci_values.clear()
    time_values.clear()
    for term in terms_list_long:
        start_time = time.perf_counter()
        value = fibonacci_dynamic(term)
        elapsed_time = time.perf_counter()
        elapsed_time = round(elapsed_time - start_time, 6)
        time_values.append(elapsed_time)
        fibonacci_values.append(value)

    plt.plot(terms_list_long,
             time_values,
             marker="o")
    plt.title("Growth of Time Complexity in Dynamic Programming Algorithm for Fibonacci Series")
    plt.xlabel("n-th term")
    plt.ylabel("Time (s)")
    plt.grid(True)
    for i in range(len(terms_list_long)):
        plt.text(terms_list_long[i],
                 time_values[i] + 0.001,
                 str(f"n={terms_list_long[i]} | ts={round(time_values[i], 12)}"),
                 ha='center',
                 bbox=dict(
                     facecolor='white',
                     edgecolor='black',
                     boxstyle='round,pad=0.5')
                 )
    plt.show()

    pretty_table.field_names = ["n"] + terms_list_long
    pretty_table.add_row(["Time(s)"] + time_values)
    pretty_table.add_row(["nth Term"] + fibonacci_values)  # TODO: fix print
    print(pretty_table)
