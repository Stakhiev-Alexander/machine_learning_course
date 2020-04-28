import sys

import bank_scoring_example
import bayes_classifier
import decision_tree
import kNN_classifier
import svm_classifier


def main():
    if (len(sys.argv) == 2):
        arg = int(sys.argv[1])
        # done
        # Task 1
        if (arg == 1):
            bayes_classifier.tic_tac_toe_example()
            bayes_classifier.spam_example()
            return
        # done
        # Task 2
        if (arg == 2):
            bayes_classifier.dots_example()
            return
        # done
        # Task 3
        if (arg == 3):
            kNN_classifier.glass_example()
            return
        # done
        # Task 5
        if (arg == 5):
            # decision_tree.glass_example_tree()
            decision_tree.spam7_example_tree()
            return
        # done
        # Task 6
        if (arg == 6):
            bank_scoring_example.bank_scoring_example()
            return

    # Task 4
    if ((len(sys.argv) == 3) and (int(sys.argv[1]) == 4)):
        arg = sys.argv[2]
        # a
        if (arg == 'a'):
            svm_classifier.a_example()
            return
        # b
        if (arg == 'b'):
            svm_classifier.b_example()
            return
        # c
        if (arg == 'c'):
            svm_classifier.c_example()
            return
        # d
        if (arg == 'd'):
            svm_classifier.d_example()
            return
        # e
        if (arg == 'e'):
            svm_classifier.e_example()
            return


if __name__ == "__main__":
    main()
