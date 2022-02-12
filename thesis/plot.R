librarian::shelf(tidyverse, ggplot2, tikzDevice, extrafont, reticulate)
options(tikzDefaultEngine ='luatex')
options(tikzDocumentDeclaration = "\\documentclass{scrreprt}")
options(tikzLualatexPackages = c(
	getOption("tikzLualatexPackages"),
	"\\setmainfont{Open Sans}"
))

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
	ungroup() %>%
	transmute(
		Compiler = compiler %>% fct_recode("Cray" = "cray", "GNU" = "gnu", "Intel" = "intel", "PGI" = "pgi"),
		Language = lang %>% fct_recode("Fortran" = "fortran", "C++" = "cpp", "Rust" = "rust"),
		Algorithm = algorithm,
		Type = type %>% fct_recode("sequential" = "seq", "CPU" = "cpu", "GPU" = "gpu"),
		Technology = technology,
		N = nx,
		`# cores` = nproc,
		`Runtime (s)` = runtime,
		Error = error,
		`Strong speedup` = strong_speedup
	)

plot_strong_runtime <- function(data, size) {
	data %>%
		filter(Type == "CPU", N == size) %>%
		ggplot(aes(x = `# cores`, y = `Runtime (s)`)) +
		facet_wrap(~ Language + Compiler, ncol = 4) +
		scale_x_continuous(trans = "log2", breaks = c(1, 2, 4, 8), minor_breaks = c(3, 6, 12)) +
		scale_y_continuous(trans = "log10") +
		geom_line(aes(linetype = Algorithm, color = Technology)) +
		theme(
			legend.justification = c(1, 0),
			legend.position = c(1, -0.14),
			legend.direction = "horizontal",
			axis.title.x = element_text(hjust = 0.068),
			axis.title.y = element_blank(),
			plot.title.position = "plot"
		) +
		guides(
			linetype = guide_legend(title.position = "top"),
			color = guide_legend(title.position = "top")
		) +
		labs(subtitle = "Runtime (s)")
}
plot_strong_speedup <- function(data, size) {
	data %>%
		filter(Type == "CPU", N == size) %>%
		arrange(`# cores`) %>%
		ggplot(aes(x = `# cores`, y = `Strong speedup`)) +
		facet_wrap(~ Language + Compiler, ncol = 4) +
		scale_x_continuous(trans = "log2", breaks = c(1, 2, 4, 8), minor_breaks = c(3, 6, 12)) +
		scale_y_continuous(trans = "log10") +
		geom_line(aes(linetype = Algorithm, color = Technology)) +
		theme(
			legend.justification = c(1, 0),
			legend.position = c(1, -0.14),
			legend.direction = "horizontal",
			axis.title.x = element_text(hjust = 0.068),
			axis.title.y = element_blank(),
			plot.title.position = "plot"
		) +
		guides(
			linetype = guide_legend(title.position = "top"),
			color = guide_legend(title.position = "top")
		) +
		labs(subtitle = "Strong speedup")
}

colors <- c(
	"#1F407A",
	"#485A2C",
	"#1269B0",
	"#72791C",
	"#91056A",
	#"#6F6F6F",
	"#A8322D",
	"#007A96",
	"#956013"
)

theme_update(
	text = element_text(family = "Open Sans"),
	legend.position = "bottom",
	legend.box = "vertical",
#	legend.title = element_blank()
)

# theme_update(
# 	legend.text = element_text(size = 40, family = "Open Sans"),
# 	axis.text.x = element_text(size = 32, family = "Open Sans"),
# 	axis.title.x = element_text(size = 40, family = "Open Sans"),
# 	axis.text.y = element_text(size = 32, family = "Open Sans"),
# 	axis.title.y = element_text(size = 40, family = "Open Sans"),
# 	plot.title = element_text(size = 64, family = "Open Sans"),
# 	strip.text.x = element_text(size = 32, family = "Open Sans"),
# 	strip.text.y = element_text(size = 32, family = "Open Sans")
# )

plot_strong_runtime(data,  128)
plot_strong_runtime(data,  256)
plot_strong_runtime(data,  512)
plot_strong_runtime(data, 1024)

plot_strong_speedup(data,  128)
ggsave("strong_scaling_cpu_128.tex", device = tikz, width = 6, height = 5, dpi = 300, sanitize = TRUE)
plot_strong_speedup(data,  256)
ggsave("strong_scaling_cpu_256.tex", device = tikz, width = 6, height = 5, dpi = 300, sanitize = TRUE)
plot_strong_speedup(data,  512)
ggsave("strong_scaling_cpu_512.tex", device = tikz, width = 6, height = 5, dpi = 300, sanitize = TRUE)
plot_strong_speedup(data, 1024)
ggsave("strong_scaling_cpu_1024.tex", device = tikz, width = 6, height = 5, dpi = 300, sanitize = TRUE)

data %>%
	filter(Type == "CPU", N / `# cores` == 128) %>%
	ggplot(aes(x = N, y = `Runtime (s)`)) +
	facet_wrap( ~ Language + Compiler, ncol = 4) +
	scale_x_continuous(trans = "log2", breaks = c(128, 256, 512, 1024)) +
	scale_y_continuous(trans = "log10") +
	geom_line(aes(linetype = Algorithm, color = Technology)) +
	theme(
		axis.text.x  = element_text(angle = 45, vjust = 0.5),
		legend.justification = c(1, 0),
		legend.position = c(1, -0.14),
		legend.direction = "horizontal",
		axis.title.x = element_text(hjust = 0.11),
		axis.title.y = element_blank(),
		plot.title.position = "plot"
	) +
	guides(
		linetype = guide_legend(title.position = "top"),
		color = guide_legend(title.position = "top")
	) +
	labs(subtitle = "Runtime (s)")
ggsave("weak_scaling_cpu.tex", device = tikz, width = 6, height = 5, dpi = 300, sanitize = TRUE)

data %>%
	filter(Type == "CPU") %>%
	filter(`# cores` == 12) %>%
	ggplot(aes(x = N, y = `Runtime (s)`)) +
	facet_wrap(~ Language + Compiler, ncol = 4) +
	scale_x_continuous(trans = "log2", breaks = c(128, 256, 512, 1024)) +
	scale_y_continuous(trans = "log10") +
	geom_line(aes(linetype = Algorithm, color = Technology)) +
	theme(
		axis.text.x  = element_text(angle = 45, vjust = 0.5),
		legend.justification = c(1, 0),
		legend.position = c(1, -0.14),
		legend.direction = "horizontal",
		axis.title.x = element_text(hjust = 0.11),
		axis.title.y = element_blank(),
		plot.title.position = "plot"
	) +
	guides(
		linetype = guide_legend(title.position = "top"),
		color = guide_legend(title.position = "top")
	) +
	labs(subtitle = "Runtime (s)")
ggsave("scaling_cpu.tex", device = tikz, width = 6, height = 5, dpi = 300, sanitize = TRUE)

data %>%
	filter(Type == "sequential") %>%
	ggplot(aes(x = N, y = `Runtime (s)`)) +
	facet_wrap(~ Language + Compiler, ncol = 4) +
	scale_x_continuous(trans = "log2", breaks = c(128, 256, 512, 1024)) +
	scale_y_continuous(trans = "log10") +
	geom_line(aes(linetype = Algorithm, color = Technology)) +
	theme(
		axis.text.x  = element_text(angle = 45, vjust = 0.5),
		legend.justification = c(1, 0),
		legend.position = c(1, -0.14),
		legend.direction = "horizontal",
		axis.title.x = element_text(hjust = 0.11),
		axis.title.y = element_blank(),
		plot.title.position = "plot"
	) +
	guides(
		linetype = guide_legend(title.position = "top"),
		color = guide_legend(title.position = "top", ncol = 4)
	) +
	labs(subtitle = "Runtime (s)")
ggsave("scaling_seq.tex", device = tikz, width = 6, height = 5, dpi = 300, sanitize = TRUE)

data %>%
	filter(Type == "sequential", Language == "Rust") %>%
	ggplot(aes(x = N, y = `Runtime (s)`)) +
	scale_x_continuous(trans = "log2", breaks = c(128, 256, 512, 1024)) +
	scale_y_continuous(trans = "log10") +
	geom_line(aes(linetype = Algorithm, color = Technology)) +
	theme(
		axis.title.y = element_blank(),
		plot.title.position = "plot"
	) +
	labs(subtitle = "Runtime (s)")
ggsave("scaling_seq_rust.tex", device = tikz, width = 6, height = 5, dpi = 300, sanitize = TRUE)

data %>%
	filter(Type == "GPU") %>%
	ggplot(aes(x = N, y = `Runtime (s)`)) +
	facet_wrap(~ Language + Compiler) +
	scale_x_continuous(trans = "log2", breaks = c(128, 256, 512, 1024)) +
	scale_y_continuous(trans = "log10") +
	geom_line(aes(linetype = Algorithm, color = Technology)) +
	theme(
		axis.title.y = element_blank(),
		plot.title.position = "plot"
	) +
	labs(subtitle = "Runtime (s)")
ggsave("scaling_gpu.tex", device = tikz, width = 6, height = 5, dpi = 300, sanitize = TRUE)

data %>%
	select(
		Type,
		Language,
		Compiler,
		Technology,
		Algorithm,
		N,
		`# cores`,
		`Runtime (s)`,
		Error
	) %>%
	arrange(N, Type, Language, Compiler, Technology, `# cores`) %>%
	mutate(
		`Runtime (s)` = `Runtime (s)` %>% format(nsmall = 9, trim = TRUE),
		Error = Error %>% format(nsmall = 9, trim = TRUE)
	) %>%
	format_delim("&", col_names = FALSE) %>%
	str_replace_all("_", "\\\\_") %>%
	str_replace_all("\n", "\\\\\\\\\n") %>%
	cat(file = "data.tex")

in_field <- np$load("../stencil/in_field_base.npy")
out_field <- np$load("../stencil/out_field_base.npy")

bind_rows(
	in_field[3:130, 3:130, 32] %>%
		as_tibble() %>%
		rownames_to_column("x") %>%
		pivot_longer(-c(x), names_to = "y", values_to = "phi") %>%
		mutate(type = "in") %>%
		mutate(
			x = x %>% as.numeric(),
			y = y %>% str_remove("^V") %>% as.numeric()
		) %>%
		mutate(
			x = x / max(x),
			y = y / max(y)
		),
	out_field[3:130, 3:130, 32] %>%
		as_tibble() %>%
		rownames_to_column("x") %>%
		pivot_longer(-c(x), names_to = "y", values_to = "phi") %>%
		mutate(type = "out") %>%
		mutate(
			x = x %>% as.numeric(),
			y = y %>% str_remove("^V") %>% as.numeric()
		) %>%
		mutate(
			x = x / max(x),
			y = y / max(y)
		)
) %>%
	mutate(
		type = type %>% as_factor() %>% fct_recode("Initial conditions" = "in", "Solution" = "out")
	) %>%
	ggplot(aes(x, y, fill = phi)) +
	facet_grid(~ type) +
	geom_raster() +
	scale_fill_continuous(guide = guide_colorbar(title = "$\\phi$", barwidth = 20)) +
	coord_equal()
ggsave("initial_conditions.tex", device = tikz, width = 6, height = 4, dpi = 300)
