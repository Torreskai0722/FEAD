-- CREATE TABLE emsdata (
--     rowkey VARCHAR(50) NOT NULL,
--     ems_value VARCHAR(300) NOT NULL,
--     ems_time VARCHAR(50) NOT NULL,
--     imei VARCHAR(50) NOT NULL,
--     gpsno VARCHAR(50) NOT NULL,
--     truckid VARCHAR(100) NOT NULL,
--     itemid VARCHAR(50) NOT NULL,
--     model VARCHAR(20) NOT NULL,
--     truckno VARCHAR(20) NOT NULL,
--     orgcode VARCHAR(20) NOT NULL,
--     lat VARCHAR(50),
--     lng VARCHAR(50),
--     course VARCHAR(50),
--     triggertime VARCHAR(50),
--     province VARCHAR(50),
--     city VARCHAR(50),
--     county VARCHAR(50),
--     address VARCHAR(200),
--     r_name VARCHAR(50),
--     r_level VARCHAR(50),
--     tdate VARCHAR(50) NOT NULL,
--     PRIMARY KEY (rowkey)
-- );

LOAD DATA INFILE 'dwa_inceptio_ems_detail_location_manual.csv' 
INTO TABLE emsdata 
FIELDS TERMINATED BY '	' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n';