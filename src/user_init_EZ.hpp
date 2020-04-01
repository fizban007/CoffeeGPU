#ifndef _USER_INIT_H_
#define _USER_INIT_H_

Stagger st_e[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
Stagger st_b[3] = {Stagger(0b111), Stagger(0b111), Stagger(0b111)};
data.E.set_stagger(st_e);
data.B.set_stagger(st_b);
data.B0.set_stagger(st_b);

data.E.initialize();
data.B.initialize();
data.B0.initialize();
data.P.assign(0.0);
data.P.sync_to_device();

#endif  // _USER_INIT_H_
