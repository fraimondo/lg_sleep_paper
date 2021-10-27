library('readxl')
library('tidyr')
library('DescTools')

data_path <- '../data/behavioral'
fname <- sprintf('%s/tables_stats.xlsx', data_path)


h_page_columns_acc <- c('subject', 'W', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'earlyN2', 'lateN2')
h_page_columns_rt <- c('subject', 'W', 'H1', 'H2', 'H3', 'H4', 'H5')
h_page_levels_acc <- c('W', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'earlyN2', 'lateN2')
h_page_levels_rt <- c('W', 'H1', 'H2', 'H3', 'H4', 'H5')

d_page_columns_acc <- c('subject', 'W', 'D1', 'D2', 'D3', 'D4', 'N2')
d_page_columns_rt <- c('subject', 'W', 'D1', 'D2', 'D3')
d_page_levels_acc <- c('W', 'D1', 'D2', 'D3', 'D4', 'N2')
d_page_levels_rt <- c('W', 'D1', 'D2', 'D3')

message('============')
message('Accuracy (D)')
message('============')

data <- read_excel(fname, sheet='behaviour_Acc_D')
data$subject <- seq.int(nrow(data))
long_df <- data %>% gather(Stage, value, W:N2)

k <- kruskal.test(value ~ Stage, data=long_df)
print(k)

page_df <- na.omit(data[, d_page_columns_acc])
long_page_df <- page_df %>% gather(Stage, value, W:N2)
long_page_df$Stage <- factor(long_page_df$Stage, levels=rev(d_page_levels_acc))
long_page_df$subject <- factor(long_page_df$subject)
p <- PageTest(value ~ Stage | subject, data=long_page_df)
print(p)


message('============')
message('Accuracy (H)')
message('============')

data <- read_excel(fname, sheet='behaviour_Acc_H')
data$subject <- seq.int(nrow(data))
long_df <- data %>% gather(Stage, value, W:lateN2)

k <- kruskal.test(value ~ Stage, data=long_df)
print(k)

page_df <- na.omit(data[, h_page_columns_acc])
long_page_df <- page_df %>% gather(Stage, value, W:lateN2)
long_page_df$Stage <- factor(long_page_df$Stage, levels=rev(h_page_levels_acc))
long_page_df$subject <- factor(long_page_df$subject)
p <- PageTest(value ~ Stage | subject, data=long_page_df)
print(p)

message('============')
message('RT (D)')
message('============')

data <- read_excel(fname, sheet='behaviour_RT_D')
data$subject <- seq.int(nrow(data))
long_df <- data %>% gather(Stage, value, W:D4)

k <- kruskal.test(value ~ Stage, data=long_df)
print(k)

page_df <- na.omit(data[, d_page_columns_rt])
long_page_df <- page_df %>% gather(Stage, value, W:D3)
long_page_df$Stage <- factor(long_page_df$Stage, levels=d_page_levels_rt)
long_page_df$subject <- factor(long_page_df$subject)
p <- PageTest(value ~ Stage | subject, data=long_page_df)
print(p)

message('============')
message('RT (H)')
message('============')

data <- read_excel(fname, sheet='behaviour_RT_H')
data$subject <- seq.int(nrow(data))
long_df <- data %>% gather(Stage, value, W:lateN2)

k <- kruskal.test(value ~ Stage, data=long_df)
print(k)

page_df <- na.omit(data[, h_page_columns_rt])
long_page_df <- page_df %>% gather(Stage, value, W:H5)
long_page_df$Stage <- factor(long_page_df$Stage, levels=h_page_levels_rt)
long_page_df$subject <- factor(long_page_df$subject)
p <- PageTest(value ~ Stage | subject, data=long_page_df)
print(p)


message('============')
message('MMN Ampl (D)')
message('============')

data <- read_excel(fname, sheet='MMN_ampl')
data$subject <- seq.int(nrow(data))
long_df <- data %>% gather(Stage, value, W:N2)

k <- kruskal.test(value ~ Stage, data=long_df)
print(k)

page_df <- na.omit(data[, d_page_columns_acc])
long_page_df <- page_df %>% gather(Stage, value, W:N2)
long_page_df$Stage <- factor(long_page_df$Stage, levels=rev(d_page_levels_acc))
long_page_df$subject <- factor(long_page_df$subject)
p <- PageTest(value ~ Stage | subject, data=long_page_df)
print(p)


message('============')
message('MMN Time (D)')
message('============')

data <- read_excel(fname, sheet='MMN_time')
data$subject <- seq.int(nrow(data))
long_df <- data %>% gather(Stage, value, W:N2)

k <- kruskal.test(value ~ Stage, data=long_df)
print(k)

page_df <- na.omit(data[, d_page_columns_acc])
long_page_df <- page_df %>% gather(Stage, value, W:N2)
long_page_df$Stage <- factor(long_page_df$Stage, levels=d_page_levels_acc)
long_page_df$subject <- factor(long_page_df$subject)
p <- PageTest(value ~ Stage | subject, data=long_page_df)
print(p)
