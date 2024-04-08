rm(list=ls(all=TRUE)) 
library(logitnorm)
library(XLConnect)
library(Rcpp)
library(REddyProc)
library(csv)

#data <- read.csv(file="G:/ML_ET/random_forest/input_fluxdata/illinois/US_1B1/US-IB1/gaps.csv",header=TRUE)




dir.create("G:/ML_ET/random_forest/input_fluxdata/forecast")

#data <- read.csv(file="G:/ML_ET/random_forest/input_fluxdata/illinois/US_Bo1/gaps.csv",header=TRUE)
#data <- read.csv(file="G:/ML_ET/random_forest/input_fluxdata/illinois/US-Bo2/gaps.csv",header=TRUE)
#data <- read.csv(file="G:/ML_ET/random_forest/input_fluxdata/indiana/US-Br1/gaps.csv",header=TRUE)
#data <- read.csv(file="G:/ML_ET/random_forest/input_fluxdata/indiana/US-Br3/gaps.csv",header=TRUE)
#data <- read.csv(file="G:/ML_ET/random_forest/input_fluxdata/michigan/MI_nonirri/gaps.csv",header=TRUE)

#data <- read.csv(file="G:/ML_ET/random_forest/input_fluxdata/nebraska/US-Ne2/gaps.csv",header=TRUE)

data <- read.csv(file="G:/ML_ET/random_forest/input_fluxdata/ohio/AMF_US-CRT_BASE-BADM_3-5/gaps.csv",header=TRUE)

#data <- read.csv(file="G:/flux_tower_data/potato/gaps.csv",header=TRUE)

oct<-(data)

oct=oct[,-1]

EddyDataWithPosix.F <- fConvertTimeToPosix(
  oct, 'YDH', Year.s = 'Year', Day.s = 'DoY', Hour.s = 'Hour')

EddySetups.C <- sEddyProc$new('oct', EddyDataWithPosix.F, c('NEE','LE','H','Rg','Tair','VPD','Ustar'))

EddyProc.C<- sEddyProc$new('oct', EddyDataWithPosix.F, c('NEE','LE','H','Rg','Tair','VPD','Ustar'))

uStarTh <- EddyProc.C$sEstUstarThreshold()$uStarTh

EddyProc.C$sMDSGapFillAfterUstar('LE')

EddyProc.C$sExportResults()

filled<- EddyProc.C$sExportResults()

filledall=data.frame(cbind(Year=EddyDataWithPosix.F$Year,DoY=EddyDataWithPosix.F$DoY,Hour=EddyDataWithPosix.F$Hour,LE=filled$LE_uStar_f))

#############################################################################

#write.csv(filledall,"G:/ML_ET/random_forest/input_fluxdata/nebraska/US-Ne2/filled.csv")

#write.csv(filledall,"G:/flux_tower_data/potato/filled.csv")

write.csv(filledall,"G:/ML_ET/random_forest/input_fluxdata/ohio/AMF_US-CRT_BASE-BADM_3-5/filled.csv")

