close all;

tic;
results = runtests('ut.insertElementTest');
toc;

tic;
results = runtests('ut.deleteElementTest');
toc;

tic;
results = runtests('ut.crpTest');
toc;

tic;
results = runtests('ut.ncrpTest');
toc;

tic;
results = runtests('ut.sbpTest');
toc;

tic;
results = runtests('ut.hsbpTest');
toc;

tic;
results = runtests('ut.sbptreeTest');
toc;

tic;
results = runtests('ut.stickbreakingTest');
toc;