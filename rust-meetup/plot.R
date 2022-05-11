librarian::shelf(tidyverse, ggplot2, showtext)
showtext_auto()

data <- "data.txt" %>%
	read_fwf(., fwf_empty(., col_names = c("compiler", "lang", "version", "nx", "ny", "nz", "num_iter", "nproc", "runtime", "error"), n = Inf), col_types = "ffciiiiidd") %>%
	mutate(
		correct = error < 2.5e-4,
		algorithm = case_when(
			str_starts(version, "laplap") ~ "laplap",
			str_starts(version, "inline") ~ "inline",
			TRUE                          ~ NA_character_
		) %>% as_factor(),
		type = case_when(
			str_detect(version, "seq")           ~ "seq",
			str_detect(version, "par")           ~ "cpu",
			str_detect(version, "openmp$")       ~ "cpu",
			str_detect(version, "openacc")       ~ "gpu",
			str_detect(version, "openmp_target") ~ "gpu",
			str_detect(version, "cuda")          ~ "gpu",
			str_detect(version, "accel")         ~ "gpu",
			TRUE                                 ~ NA_character_
		) %>% as_factor(),
		technology = str_remove(version, "^(laplap|inline)_")
	) %>%
	group_by(compiler, lang, version, nx, ny) %>%
	arrange(nproc) %>%
	mutate(
		strong_speedup = first(runtime) / runtime
	) %>%
	ungroup()

plot_strong_runtime <- function(data, size) {
	data %>%
		filter(type == "cpu") %>%
		filter(nx == size, ny == size) %>%
		ggplot(aes(x = nproc, y = runtime)) +
		ggtitle(str_interp("Strong scaling ${size}x${size}x64")) +
		facet_grid(lang ~ compiler) +
		scale_x_continuous(trans = "log2", breaks = c(1, 2, 4, 8), minor_breaks = c(3, 6, 12)) +
		scale_y_continuous(trans = "log10") +
		geom_line(aes(linetype = algorithm, color = technology))
}
plot_strong_speedup <- function(data, size) {
	data %>%
		filter(type == "cpu") %>%
		filter(nx == size, ny == size) %>%
		ggplot(aes(x = nproc, y = strong_speedup)) +
		ggtitle(str_interp("Strong scaling ${size}x${size}x64")) +
		facet_grid(lang ~ compiler) +
		scale_x_continuous(trans = "log2", breaks = c(1, 2, 4, 8), minor_breaks = c(3, 6, 12)) +
		scale_y_continuous(trans = "log10") +
		geom_line(aes(linetype = algorithm, color = technology))
}

theme_update(
	legend.position = "bottom",
	legend.title = element_blank(),
	legend.text = element_text(size = 20),
	axis.text.x = element_text(size = 16),
	axis.title.x = element_text(size = 20),
	axis.text.y = element_text(size = 16),
	axis.title.y = element_text(size = 20),
	plot.title = element_text(size = 32),
	strip.text.x = element_text(size = 16),
	strip.text.y = element_text(size = 16)
)

plot_strong_runtime(data,  128)
plot_strong_runtime(data,  256)
plot_strong_runtime(data,  512)
plot_strong_runtime(data, 1024)

plot_strong_speedup(data,  128)
ggsave("strong_scaling_cpu_128.png", width = 6, height = 4, dpi = 300)
plot_strong_speedup(data,  256)
ggsave("strong_scaling_cpu_256.png", width = 6, height = 4, dpi = 300)
plot_strong_speedup(data,  512)
ggsave("strong_scaling_spu_512.png", width = 6, height = 4, dpi = 300)
plot_strong_speedup(data, 1024)
ggsave("strong_scaling_cpu_1024.png", width = 6, height = 4, dpi = 300)

data %>%
	filter(type == "cpu") %>%
	filter(nx / nproc == 128) %>%
	ggplot(aes(x = nx, y = runtime)) +
	ggtitle("CPU: weak scaling") +
	facet_grid(lang ~ compiler) +
	scale_x_continuous(trans = "log2", breaks = c(128, 256, 512, 1024)) +
	scale_y_continuous(trans = "log10") +
	geom_line(aes(linetype = algorithm, color = technology)) +
	theme(axis.text.x  = element_text(angle = 45, vjust = 0.5))
ggsave("weak_scaling_cpu.png", width = 6, height = 4, dpi = 300)

data %>%
	filter(type == "seq") %>%
	ggplot(aes(x = nx, y = runtime)) +
	ggtitle("Sequential: weak scaling") +
	facet_grid(lang ~ compiler) +
	scale_x_continuous(trans = "log2", breaks = c(128, 256, 512, 1024)) +
	scale_y_continuous(trans = "log10") +
	geom_line(aes(linetype = algorithm, color = technology)) +
	theme(axis.text.x  = element_text(angle = 45, vjust = 0.5))
ggsave("weak_scaling_seq.png", width = 6, height = 4, dpi = 300)

data %>%
	filter(type == "gpu") %>%
	ggplot(aes(x = nx, y = runtime)) +
	ggtitle("GPU: weak scaling") +
	facet_wrap(~ lang + compiler) +
	scale_x_continuous(trans = "log2", breaks = c(128, 256, 512, 1024)) +
	scale_y_continuous(trans = "log10") +
	geom_line(aes(linetype = algorithm, color = technology))
ggsave("weak_scaling_gpu.png", width = 6, height = 4, dpi = 300)
