%%%%%
%% Specifies the \LaTeX\ engine.
%%%%%
\NeedsTeXFormat{LaTeX2e}

%%%%%
%% Provides the name of the class.
%%%%%
\ProvidesClass{jbsc}

%%%%%
%% Populates defaults from the `article` class.
%%%%%
\LoadClass[12pt]{article}

%%%%%
%% Requires certain packages.
%%%%%

%%%
%% Loading the `nag` package to check for depreciated syntax.
%%%
\RequirePackage[l2tabu, orthodox]{nag}

%%%
%% Fonts and Typewriting
%%%
\RequirePackage{mathtools}
\RequirePackage{amssymb}
\RequirePackage{bbm}
\RequirePackage{bm}
\RequirePackage{lmodern}
\RequirePackage{nicefrac}
\RequirePackage{relsize}
\setlength{\jot}{1em}

%%%
%% Paragraph Formatting
%%%
\RequirePackage{setspace}
\RequirePackage{indentfirst}
\RequirePackage{hanging}
\RequirePackage{ragged2e}

%%%
%% Document size and margins
%%%
\RequirePackage[a4paper, margin = 2.5cm]{geometry}
\RequirePackage{fancyhdr}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\cfoot{\thepage}
\pagestyle{fancy}

%%%
%% Language Support
%%%
\RequirePackage[english]{babel}
\RequirePackage{csquotes}

%%%
%% Author Block
%%%
\RequirePackage{authblk}

%%%
%% Enables including images
%%%
\RequirePackage{graphicx}
\graphicspath{{figures/}}

%%%
%% Enhances Figures and Tables environments
%%%
\RequirePackage{placeins}
\RequirePackage[hypcap=true]{caption}
\captionsetup[table]{labelfont=bf,textfont=it}
\captionsetup[figure]{labelfont=bf,textfont=it}
\RequirePackage{booktabs, multirow, array,tabularx, floatrow,                   cellspace, threeparttable}
\floatsetup[table]{capposition = top}
\newcolumntype{Z}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
\renewcommand{\arraystretch}{1.25}

%%%
%% Enables hyperlinks
%%%
\RequirePackage{varioref}
\RequirePackage{hyperref}
\hypersetup{
    colorlinks = true,
    linkcolor  = black,
    filecolor  = black,      
    urlcolor   = black,
    citecolor  = black
    }
 
\urlstyle{same}

%%%
%% Cross References
%%%
\RequirePackage{cleveref}

%%%
%% Citation Manager
%%%
\RequirePackage[
    authordate,
    natbib,
    backend		 = biber,
    sorting		 = nyt,
    backref      = true,
    backrefstyle = three+,
    ibidtracker  = constrict
]{biblatex-chicago}

\addbibresource{bibliography.bib}