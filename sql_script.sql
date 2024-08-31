-- Table: public.technical_data

-- DROP TABLE IF EXISTS public.technical_data;

CREATE TABLE IF NOT EXISTS public.technical_data
(
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.technical_data
    OWNER to postgres;

COMMENT ON TABLE public.technical_data
    IS 'Table containing relevant technical data';
	
ALTER TABLE IF EXISTS public.technical_data
	ADD tech_key int primary key,
	ADD momentum_kama numeric,
	ADD others_cr numeric,
	ADD trend_ema_fast numeric,
	ADD trend_ema_slow numeric,
	ADD trend_ichimoku_a numeric,
	ADD trend_ichimoku_b numeric,
	ADD trend_ichimoku_base numeric,
	ADD trend_ichimoku_conv numeric,
	ADD trend_sma_fast numeric,
	ADD trend_sma_slow numeric,
	ADD trend_visual_ichimoku_a numeric,
	ADD trend_visual_ichimoku_b numeric,
	ADD volatility_bbh numeric,
	ADD volatility_bbl numeric,
	ADD volatility_bbm numeric,
	ADD volatility_dch numeric,
	ADD volatility_dcl numeric,
	ADD volatility_dcm numeric,
	ADD volatility_kcc numeric,
	ADD volatility_kch numeric,
	ADD volatility_kcl numeric,
	ADD volume_obv numeric,
	ADD volume_vpt numeric,
	ADD volume_vwap numeric,
	ADD date_time timestamp;
	
	
CREATE TABLE IF NOT EXISTS PUBLIC.financial_data(
	fin_key int PRIMARY KEY,
	"open" numeric,
	high numeric,
	low numeric,
	"close" numeric,
	tick_vol numeric,
	vol numeric,
	spread numeric,
	date_time timestamp
);
	


