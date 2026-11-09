u1=0.05
du1=2*u1*2/3
du2=2*u1*1/3
u(x)=x<0.5?(-u1+du1*x*2):(u1-du2*2*(1-x))
set xrange [0:1]
#set yrange [-0.05:0.05]
set xlabel '$x/l_0$'
set ylabel '$u_x$'
set grid
plot 'ux.dat' using (($0)/512.0):($1) t 'расчет', u(x) lw 3 t 'теория'
