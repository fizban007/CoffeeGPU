#ifndef _USER_INIT_H_
#define _USER_INIT_H_

Scalar a0 = 0.0;

Stagger st_e[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
Stagger st_b[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
data.E.set_stagger(st_e);
data.B.set_stagger(st_b);
data.B0.set_stagger(st_b);

data.B.initialize(0, [&](Scalar x, Scalar y, Scalar z) {
	return 0.0;  
});

data.B.initialize(1, [&](Scalar x, Scalar y, Scalar z) {
	return 0.0;  
});

data.B.initialize(2, [&](Scalar x, Scalar y, Scalar z) {
	return 1.0 / get_sqrt_gamma(a0, x, y, z);  
});


data.E.initialize();
data.B0.initialize();
data.P.assign(0.0);
data.P.sync_to_device();

#endif  // _USER_INIT_H_