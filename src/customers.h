#ifndef CUSTOMERS_H
#define CUSTOMERS_H

#include <sorach/sorach.h>

#define N_CUSTOMERS 5
#define N_FEATURES 62

static const SorScalar customers[N_CUSTOMERS * (N_FEATURES + 1)] = {
    // Kate: retired, previous campaign succeeded, called in October
    //   age 67, call 420 s, contacts this campaign 1, last contacted 6 days ago, previous contacts 2
    //   retired, married, basic.4y, default no, housing loan no, personal loan no, cellular, oct tue, previous outcome success
    1, 67, 420, 1, 6, 2, -3.4, 92.431, -26.9, 0.739, 5017.5, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 1,

    // Jessie: student, long call in March
    //   age 23, call 610 s, contacts this campaign 1, not contacted before, previous contacts 0
    //   student, single, high.school, default no, housing loan yes, personal loan no, cellular, mar thu, previous outcome nonexistent
    1, 23, 610, 1, 999, 0, -1.8, 93.369, -34.8, 0.646, 5008.7, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 1, 0,

    // Lewis: blue-collar worker, short call in May, contacted many times
    //   age 38, call 45 s, contacts this campaign 6, not contacted before, previous contacts 0
    //   blue-collar, married, basic.9y, default unknown, housing loan yes, personal loan yes, telephone, may mon, previous outcome nonexistent
    1, 38, 45, 6, 999, 0, 1.1, 93.994, -36.4, 4.857, 5191, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 0,

    // Steve: manager, medium-length call in August
    //   age 45, call 280 s, contacts this campaign 2, not contacted before, previous contacts 1
    //   management, divorced, university.degree, default no, housing loan yes, personal loan no, cellular, aug wed, previous outcome failure
    1, 45, 280, 2, 999, 1, 1.4, 93.444, -36.1, 4.964, 5228.1, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
    0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    1, 0, 0,

    // Robert: technician, average customer in June
    //   age 34, call 200 s, contacts this campaign 3, not contacted before, previous contacts 0
    //   technician, single, professional.course, default no, housing loan no, personal loan no, telephone, jun fri, previous outcome nonexistent
    1, 34, 200, 3, 999, 0, 1.4, 94.465, -41.8, 4.961, 5228.1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 1, 0
};

static const char *customers_names[N_CUSTOMERS] = {
    "Kate", "Jessie", "Lewis", "Steve", "Robert"
};

static const char *customers_details[N_CUSTOMERS] = {
    "retired, previous campaign succeeded, called in October\nage 67, call 420 s, contacts this campaign 1, last contacted 6 days ago, previous contacts 2",

    "student, long call in March\nage 23, call 610 s, contacts this campaign 1, not contacted before, previous contacts 0",

    "blue-collar worker, short call in May, contacted many times\nage 38, call 45 s, contacts this campaign 6, not contacted before, previous contacts 0",

    "manager, medium-length call in August\nage 45, call 280 s, contacts this campaign 2, not contacted before, previous contacts 1",

    "technician, average customer in June\nage 34, call 200 s, contacts this campaign 3, not contacted before, previous contacts 0"
};

#endif
