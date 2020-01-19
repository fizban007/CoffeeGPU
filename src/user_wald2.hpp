#ifndef _USER_INIT_H_
#define _USER_INIT_H_

Scalar a0 = 0.0;

data.B.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
	return 0.0;  
});

data.B.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
	return 0.0;  
});

data.B.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
	return 1.0 / get_sqrt_gamma(a0, x, y, z);  
});

data.B.sync_to_device();

data.E.initialize();
data.E.sync_to_device();

data.B0.initialize();
data.B0.sync_to_device();

#endif  // _USER_INIT_H_