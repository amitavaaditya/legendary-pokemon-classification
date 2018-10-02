import psycopg2
import pandas as pd


def db_init():
    conn = psycopg2.connect(database='chillindo', user='techno', password='')
    curr = conn.cursor()
    return conn, curr


def init_tables(conn, curr):
    drop_sql = 'drop table if exists pokemon'
    curr.execute(drop_sql)
    create_sql = '''
    CREATE TABLE pokemon(
        _id serial primary key,
        index integer not null,
        name varchar(30) not null,
        type_1 varchar(10) not null,
        type_2 varchar(10),
        total integer not null,
        hp integer not null,
        atk integer not null,
        def integer not null,
        sp_atk integer not null,
        sp_def integer not null,
        speed integer not null,
        generation integer not null,
        legendary bool not null
    )
    '''
    curr.execute(create_sql)
    conn.commit()


def read_dataset():
    df = pd.read_excel('pokemon.xlsx', sheet_name=0, header=0)
    return df


def store_into_db(df, conn, curr):
    insert_sql = 'INSERT INTO pokemon VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
    curr.executemany(insert_sql, df.reset_index().values)
    conn.commit()


if __name__ == '__main__':
    conn, curr = db_init()
    init_tables(conn, curr)
    df = read_dataset()
    store_into_db(df, conn, curr)
