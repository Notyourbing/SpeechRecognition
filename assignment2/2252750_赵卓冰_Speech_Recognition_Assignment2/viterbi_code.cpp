#include <iostream>
#include <vector>
using namespace std;

// Number of states
const int STATE_NUM = 3;
const int DAY_NUM = 5;
// Transition probabilities between any two states
const double a[STATE_NUM][STATE_NUM] = { {0.2, 0.3, 0.5},
                                       {0.2, 0.2, 0.6},
                                       {0.0, 0.2, 0.8 } };
// Emission probabilities
const double b[STATE_NUM][STATE_NUM] = { {0.7, 0.2, 0.1},
                                       {0.3, 0.4, 0.3},
                                       {0.0, 0.1, 0.9 } };
// A C B A C
const int observation[DAY_NUM] = { 0, 2, 1, 0, 2 };
// The table
double V[DAY_NUM][STATE_NUM] = { 0 };
int backtrace_pointer[DAY_NUM][STATE_NUM];

// Initiallization
void InitV() {
  int day = 0;
  int state = observation[day];
  for (int j = 0; j < STATE_NUM; ++j) {
    V[day][j] = b[j][state] / STATE_NUM;
  }
}

void DisplayTable() {
  cout << "Table result is:" << endl;
  cout << "             A                   C                  B                  A                  C" << endl;
  for (int j = 0; j < STATE_NUM; ++j) {
    if (0 == j) {
      printf("good:    ");
    }
    else if (1 == j) {
      printf("neutral: ");
    }
    else {
      printf("bad:     ");
    }
    for (int day = 0; day < DAY_NUM; ++day) {
      printf("%9.8lf    |    ", V[day][j]);
    }
    printf("\n");
  }
}

void DisplayBacktracePointer() {
  cout<< endl << "Backtrace pointer result is:" << endl;
  for (int j = 0; j < STATE_NUM; ++j) {
    for (int day = 0; day < DAY_NUM; ++day) {
      printf("%4d", backtrace_pointer[day][j]);
    }
    printf("\n");
  }
}
void ViterbiAlgorithm() {
  InitV();
  memset(backtrace_pointer, -1, sizeof(backtrace_pointer));
  for (int day = 1; day < DAY_NUM; ++day) {
    int state = observation[day];
    for (int j = 0; j < STATE_NUM; ++j) {
      double max_value = -1;
      for (int i = 0; i < STATE_NUM; ++i) {
        if (V[day - 1][i] * a[i][j] > max_value) {
          max_value = V[day - 1][i] * a[i][j];
          backtrace_pointer[day][j] = i; // update backtrace pointer
        }
      }
      V[day][j] = b[j][state] * max_value;
    }
  }
}



int main() {
  ViterbiAlgorithm();
  DisplayTable();
  DisplayBacktracePointer();

  return 0;
}