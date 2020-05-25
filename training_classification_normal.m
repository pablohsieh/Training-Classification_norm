close all
clear all

%Probabilidades a priori(pap) de las clases
p_w_1 = 0.5; 
p_w_3 = 0.5;

% Carga de las muestras
% x_w_i(:,j) es la muestra x_j de la clase i 
x_w_1 =[-5.0100   -8.1200   -3.6800; %muestra 1
        -5.4300   -3.4800   -3.5400; %muestra 2
         1.0800   -5.5200    1.6600; %muestra 3
         0.8600   -3.7800   -4.1100; %muestra 4
        -2.6700    0.6300    7.3900; %muestra 5
         4.9400    3.2900    2.0800; %muestra 6
        -2.5100    2.0900   -2.5900; %muestra 7
        -2.2500   -2.1300   -6.9400; %muestra 8
         5.5600    2.8600   -2.2600; %muestra 9
         1.0300   -3.3300    4.3300];%muestra 10  
x_w_3 = [5.3500    2.2600    8.1300;
         5.1200    3.2200   -2.6600;
        -1.3400   -5.3100   -9.8700;
         4.4800    3.4200    5.1900;
         7.1100    2.3900    9.2100;
         7.1700    4.3300   -0.9800;
         5.7500    3.9700    6.6500;
         0.7700    0.2700    2.4100;
         0.9000   -0.4300   -8.7100;
         3.5200   -0.3600    6.4300];    
     
[mu_1,sigma_1] = param_gauss(x_w_1);
[mu_3,sigma_3] = param_gauss(x_w_3);

figure();
plot(x_w_1(:,1),x_w_1(:,2),'x','markersize',10);
hold on;
plot(x_w_3(:,1),x_w_3(:,2),'o','markersize',10);
grid on;
xlabel('x_1','FontSize',15);
ylabel('x_2','FontSize',15);
legend('w_1','w_3','location','southeast','fontSize',15);
%print(sprintf('fig_muestras_x1_vs_x2'),'-dpng');    
     
%Graficos de las distribuciones para cada feature de w_j
x=-150:0.1:150;

%Distribuciones y_w_ij donde i:feature x_i, j:clase w_j
y_w_11 = normpdf(x,mu_1(1),sigma_1(1,1)); %Distribucion normal univariable de la clase w_1 con el feature x_1
y_w_13 = normpdf(x,mu_3(1),sigma_3(1,1)); %Distribucion normal univariable de la clase w_3 con el feature x_1
y_w_21 = normpdf(x,mu_1(2),sigma_1(2,2)); %Distribucion normal univariable de la clase w_1 con el feature x_2
y_w_23 = normpdf(x,mu_3(2),sigma_3(2,2)); %Distribucion normal univariable de la clase w_3 con el feature x_2
y_w_31 = normpdf(x,mu_1(3),sigma_1(3,3)); %Distribucion normal univariable de la clase w_1 con el feature x_3
y_w_33 = normpdf(x,mu_3(3),sigma_3(3,3)); %Distribucion normal univariable de la clase w_3 con el feature x_3

figure();
plot(x,y_w_11,'-b'); %Clase 1 muestra x_1
hold on;
plot(x,y_w_13,'--b'); %Clase 3 muestra x_1
grid on;
xticks([-60:10:60]);
axis([-60 60 0 0.06])
legend('x_1 de w_1','x_1 de w_3','fontsize',14,'location','northeast');
%print(sprintf('fig_ej_comp_dist_x1'),'-dpng');

figure();
hold on;
plot(x,y_w_21,'-r'); %Clase 1 muestra x_2
plot(x,y_w_23,'--r'); %Clase 3 muestra x_2
grid on;
xticks([-60:10:60]);
axis([-60 60 0 0.06]);
legend('x_2 de w_1','x_2 de w_3','fontsize',14,'location','northeast');
%print(sprintf('fig_ej_comp_dist_x2'),'-dpng');

figure();
hold on;
plot(x,y_w_31,'-k'); %Clase 1 muestra x_3
plot(x,y_w_33,'--k'); %Clase 3 muestra x_3
grid on;
xticks([-150:25:150]);
axis([-150 150 0 0.025]);
legend('x_3 de w_1','x_3 de w_3','fontsize',14,'location','northeast');
%print(sprintf('fig_ej_comp_dist_x3'),'-dpng');

%ERROR DE APRENDIZAJE
file_error_aprendizaje = fopen('ej_cap2_resultados_aprendizaje.txt', 'w'); 
%Imprimo los resultados del error de aprendizaje todo en un .txt

fprintf(file_error_aprendizaje,'Error de aprendizaje\n');
fprintf(file_error_aprendizaje,'Se definio g = disc w_1 - disc w_3\n\n');
%Veo como clasifica a las muestras de cada feature x_k en la clase w_1
nook = 0;
clase = 1;
for feature=1:3
    fprintf(file_error_aprendizaje,'Analisis de aprendizaje para x_%d de w_%d:\n',feature,clase);
    nook = 0; %contador de no aciertos
    sample_train = x_w_1(:,feature); %cargo las muestras de la clase w_1
    for i = 1:length(sample_train)
        sample_string=sprintf('%.2f', sample_train(i));%Para imprimir por pantalla
        j = clase_1(file_error_aprendizaje,1,sample_train(i),mu_1(feature),sigma_1(feature,feature),p_w_1,mu_3(feature),sigma_3(feature,feature),p_w_3,sample_string);
            %La func clase_1 devuelve la cant de veces clasificado en el 1er parametro.
            %el primer parametro de la funcion es d=1 porque analizo 1 solo feature
            nook = nook + j; %aca tengo cant de veces bien clasificado
    end
    error = 100-100*(nook/length(sample_train));
    fprintf(file_error_aprendizaje,'El error de aprendizaje para el feature x_%d de w_%d es del %d%%\n',feature,clase,error);
    %Veo el error dado por Bhattacharyya
    [e_bhatt, p_error]= bhattacharyya(mu_1(1:feature),sigma_1(1:feature,1:feature),p_w_1,mu_3(1:feature),sigma_3(1:feature,1:feature),p_w_3);
    fprintf(file_error_aprendizaje,'La cota P(error) dada por Bhattacharyya es <= %.4f\n\n',p_error);
end

%Veo como clasifica a las muestras del feature x_k en la clase w_3
nook = 0; %contador de no aciertos
clase = 3; 
for feature=1:3
    fprintf(file_error_aprendizaje,'Analisis de aprendizaje para x_%d de w_%d:\n',feature,clase);
    nook = 0;
    sample_train = x_w_3(:,feature); %Cargo las muestras de la clase w_3
    for i = 1:length(sample_train)
        sample_string=sprintf('%.2f', sample_train(i)); %unicamente para imprimir por pantalla
        j = clase_1(file_error_aprendizaje,1,sample_train(i),mu_1(feature),sigma_1(feature,feature),p_w_1,mu_3(feature),sigma_3(feature,feature),p_w_3,sample_string);
            %La func clasif_w1 devuelve la cant de veces clasificado en el 1er parametro.
            %el primer parametro de la funcion es d=1 porque analizo 1 solo feature
        nook = nook + j; 
    end
    error = 100*(nook/length(sample_train));
    fprintf(file_error_aprendizaje,'El error de aprendizaje para el feature x_%d de w_%d es del %d%%\n',feature,clase,error);
    %Veo el error dado por Bhattacharyya
    [e_bhatt, p_error]= bhattacharyya(mu_1(1:feature),sigma_1(1:feature,1:feature),p_w_1,mu_3(1:feature),sigma_3(1:feature,1:feature),p_w_3);
    fprintf(file_error_aprendizaje,'La cota P(error) dada por Bhattacharyya es <= %.4f\n\n',p_error);
end

%CLASIFICADOR
%sample = x_w_3(1,1:d); %a modo de ejemplo para analizar se usaron las muestras de w_3
% segun d, el discriminante sera, depende los features a analizar
% si d=1 x_1
% si d=2 x_1 y x_2
% si d=3 x_1, x_2 y x_3
% no se logro de forma elegante realizar para x_1 y x_3 o x_2 y x_3 sin hardcodeo

file_clasificador = fopen('ej_cap2_resultados_clasif.txt', 'w');
%Imprimo los resultados del clasificador todo en un .txt
fprintf(file_clasificador,'Clasificacion de una muestra\n');  
fprintf(file_clasificador,'Se definio g = disc w_1 - disc w_3\n\n');
for d=1:3 
%El error de bhattacharyya depende de la cant de features a analizar
    [e_bhatt(d), p_error(d)]= bhattacharyya(mu_1(1:d),sigma_1(1:d,1:d),p_w_1,mu_3(1:d),sigma_3(1:d,1:d),p_w_3);
    fprintf(file_clasificador,'La cota P(error) dada por Bhattacharyya para %d features es <= %.4f\n',d,p_error(d));
end 
fprintf(file_clasificador,'\n');
for m=1:5
    sample = normrnd(25*(rand()-0.5),3*rand(),[1,3]);
    fprintf(file_clasificador,'%d) Dada una muestra normal aleatoria: (%.2f, %.2f, %.2f)\n',m,sample(1),sample(2),sample(3)); 
    for d=1:3
        sample_string=sprintf('%.2f, ', sample(1:d)); %para imprimir por pantalla
        fprintf(file_clasificador,'   Analisis con %d features:\n         ',d);
        nook = clase_1(file_clasificador,d,sample(1:d),mu_1(1:d),sigma_1(1:d,1:d),p_w_1,mu_3(1:d),sigma_3(1:d,1:d),p_w_3,sample_string);
    %La func clase_1 devuelve la cant de veces clasificado en el 1er parametro.
    %nook = 0 => es de w_3
    %nook = 1 => es de w_1
    end
end


fclose('all');

%%
function disc = discriminante(d,muestra,mu,sigma,pap) 
% pap:probabilidad a priori 
% d:cant de features a analizar
    disc = -0.5 * (muestra-mu)* (inv(sigma)) * (muestra-mu)' - (d/2)*log(2*pi) - 0.5*log(det(sigma)) + log(pap); 
    %Se pone primero sin trasponer porque matlab tiene vectores fila
    %No hay problemas de dimension con esta funcion porque voy a pasar los
    %datos bien de antemano
end
% --------------------------------------------

function y = gaussiana(x,mu,sigma_cuad)
    for i=1:length(x)
        y(i) = (1/sqrt(sigma_cuad*2*pi))*(exp(-(((x(i)-mu(i)).^2)/(2*sigma_cuad))));
    end
end
% --------------------------------------------

function [mu,sigma] = param_gauss(muestra)
% Obtencion de parametros gaussianos: media y matriz de cov entre features
% A partir de los datos de muestra voy a ''entrenar'' mi dicotomizador 
% esto es para ver cuales son las distribuciones de w_1 y w_3     
    mu = mean(muestra); %Media muestral
    sigma = zeros(3); %son 3 features x_1, x_2 y x_3
    for i=1:3 
        for j=1:3
            sigma(i,j) = sigma(i,j) + ( muestra(:,i) - mu(i))' * ( muestra(:,j)- mu(j)); 
            % covarianza muestral
        end
    end
    sigma = sigma / length(muestra(:,i)); 
end
% --------------------------------------------

function clasif_w1 = clase_1(file_name,d,muestra,mu1,var1,pap1,mu2,var2,pap2,cadena)
% Esta funcion devuelve la cant de veces clasificado en el param mu1,var1,pap1
    disc_1 = discriminante(d,muestra,mu1,var1,pap1);
    disc_3 = discriminante(d,muestra,mu2,var2,pap2);
    g = disc_1 - disc_3;
    if( g > 0)
        fprintf(file_name,'La muestra=(%s) pertenece a la clase w_1, porque g = %.2f\n',cadena,g);
        clasif_w1 = 1;
    else
        fprintf(file_name,'La muestra=(%s) pertenece a la clase w_3, porque g = %.2f\n',cadena,g);
        clasif_w1 = 0;
    end
end
% --------------------------------------------

function [k_bhat, p_error]= bhattacharyya(mu1,sigma1,pap1,mu2,sigma2,pap2) 
%Analisis del error acotado por Bhattacharyya bound
    k_bhat = (1/8) * (mu2-mu1) * inv( (sigma1 + sigma2)/2 ) * (mu2-mu1)' + 0.5*log(det(( (sigma1 + sigma2)/2 ))/sqrt((det(sigma1)*det(sigma2))));
    %Se pone primero sin trasponer porque matlab tiene vectores fila    
    p_error = sqrt(pap1*pap2)*(exp(-k_bhat));
    %cota de error minima dada por bhattacharyya
end