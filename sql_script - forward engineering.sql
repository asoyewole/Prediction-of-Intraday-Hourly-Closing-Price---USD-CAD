-- This script was generated by the ERD tool in pgAdmin 4.
-- Please log an issue at https://github.com/pgadmin-org/pgadmin4/issues/new/choose if you find any bugs, including reproduction steps.
BEGIN;


CREATE TABLE IF NOT EXISTS public.financial_data
(
    fin_key serial NOT NULL,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    tick_vol numeric,
    date_time timestamp without time zone,
    hist_close numeric,
    CONSTRAINT financial_data_pkey PRIMARY KEY (fin_key)
);

CREATE TABLE IF NOT EXISTS public.technical_data
(
    tech_key serial NOT NULL,
    momentum_kama numeric,
    others_cr numeric,
    trend_ema_fast numeric,
    trend_ema_slow numeric,
    trend_ichimoku_a numeric,
    trend_ichimoku_b numeric,
    trend_ichimoku_base numeric,
    trend_ichimoku_conv numeric,
    trend_sma_fast numeric,
    trend_sma_slow numeric,
    trend_visual_ichimoku_a numeric,
    trend_visual_ichimoku_b numeric,
    volatility_bbh numeric,
    volatility_bbl numeric,
    volatility_bbm numeric,
    volatility_dch numeric,
    volatility_dcl numeric,
    volatility_dcm numeric,
    volatility_kcc numeric,
    volatility_kch numeric,
    volatility_kcl numeric,
    volume_obv numeric,
    volume_vpt numeric,
    volume_vwap numeric,
    date_time timestamp without time zone,
    CONSTRAINT technical_data_pkey PRIMARY KEY (tech_key)
);

COMMENT ON TABLE public.technical_data
    IS 'Table containing relevant technical data';

CREATE TABLE IF NOT EXISTS public.transformed_date
(
    date_key serial NOT NULL,
    month_sin numeric,
    month_cos numeric,
    hour_sin numeric,
    hour_cos numeric,
    day_sin numeric,
    day_cos numeric,
    day_of_week_sin numeric,
    day_of_week_cos numeric,
    date_time timestamp without time zone,
    PRIMARY KEY (date_key)
);
END;






DELETE FROM PUBLIC.technical_data;

ALTER TABLE PUBLIC.financial_data
	ADD hist_close numeric;
	
SELECT * FROM financial_data LIMIT 10;

SELECT date_time from public.financial_data
	ORDER BY date_time DESC
	LIMIT 1;

SELECT date_time, 
		open, 
		high, 
		low, 
		close, 
		tick_vol, 
		hist_close from public.financial_data
	ORDER BY date_time DESC
	LIMIT 30;
select * from financial_data limit 10;
select * from technical_data limit 10;
drop table public.transformed_date;

CREATE TABLE IF NOT EXISTS public.transformed_date(
	month_sin decimal,
	month_cos decimal,
	hour_sin decimal,
	hour_cos decimal,
	day_sin decimal,
	day_cos decimal,
	day_of_week_sin decimal,
	day_of_week_cos decimal,
	date_time timestamp
);
ALTER TABLE public.financial_data
	DROP COLUMN vol,
	DROP COLUMN spread;
	
select count(fin_key) from public.financial_data;
select count(tech_key) from public.technical_data;
select count(date_key) from public.transformed_date;

select * from public.financial_data
order by date_time desc
limit 10;

SELECT fin_key 
FROM financial_data
WHERE date_time = '2024-08-29 23:00:00';


SELECT fin_key
	FROM financial_data
	WHERE date_time = '2024-08-26 23:00:00';
DROP SEQUENCE IF EXISTS pred_seq;

CREATE sequence pred_seq 
	START WITH 100986
	INCREMENT BY 1;


CREATE TABLE IF NOT EXISTS public.predictions(
	pred_key INT DEFAULT nextval('pred_seq') PRIMARY KEY, 
	date_time timestamp,
	predictions decimal
);


select * from public.financial_data order by date_time desc limit 5;
