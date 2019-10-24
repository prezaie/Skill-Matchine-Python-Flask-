
import pandas as pd
from sqlalchemy.engine import create_engine

def data_reader():
    """Reads data from SQL database and transform data to a DataFrame

    Args:
        none
    Returns:
        df (Dataframe): Dataframe contains employee skills and employee id

    """
    engine = create_engine('postgresql://parisarezaie@localhost/kapa') 
    df = pd.read_sql('''SELECT 
                                   name,
                                   level,
                                   project_employee_id
                            FROM criteria, employee_criteria 
                            WHERE (criteria.id = employee_criteria.criteria_id) AND (criteria.type = 'SKILL')
                            ''', engine)

    df = df.pivot(index='project_employee_id', columns='name', values='level')
    df.fillna(0,inplace=True)
    df = df.reset_index() 
    return df                    