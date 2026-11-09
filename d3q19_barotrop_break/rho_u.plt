#русский
set grid
set yrange [0:1.1]
set xrange [0:120]
set xlabel '$x$, индекс'
set ylabel '$\rho$'
plot 'rho_u.dat' using 0:1 with linespoints t '', 'rho_u.dat' using 0:3 with l t '' #, 'rho_u.dat' using 0:(($3-$1)*10)
#plot 'rho_u.dat' using 0:2 with linespoints t '', 'rho_u.dat' using 0:4 with l t ''
