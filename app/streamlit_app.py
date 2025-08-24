import os
import pathlib
import streamlit as st
import pandas as pd
import psycopg

# Basic page config
st.set_page_config(
    page_title="Literature OS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configure Streamlit paths for HF Space
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if os.environ.get("HOME", "/") in ("/", "", None):
    os.environ["HOME"] = str(PROJECT_ROOT)
os.environ.setdefault("XDG_CONFIG_HOME", os.environ["HOME"])
os.makedirs(os.path.join(os.environ["HOME"], ".streamlit"), exist_ok=True)

# Database connection
@st.cache_resource
def get_db():
    """Get database connection"""
    try:
        return psycopg.connect(
            os.environ["DB_READER_DSN"],
            application_name="literature-os-streamlit"
        )
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        raise

@st.cache_data(ttl=300)
def get_metadata(_conn):
    """Load tags and years from database"""
    try:
        with _conn.cursor() as cur:
            # Get tags
            cur.execute("SELECT DISTINCT tag FROM tags ORDER BY tag")
            tags = [r[0] for r in cur.fetchall() if r[0]]
            
            # Get years
            cur.execute("""
                SELECT DISTINCT year 
                FROM papers 
                WHERE year IS NOT NULL AND year BETWEEN 1900 AND 2100
                ORDER BY year DESC
            """)
            years = [r[0] for r in cur.fetchall() if r[0]]
            
        return tags or [], years or list(range(2010, 2026))
    except Exception as e:
        st.error(f"Failed to load metadata: {str(e)}")
        return [], list(range(2010, 2026))

@st.cache_data(ttl=300)
def get_papers(_conn, tags=None, year_start=None, year_end=None, search=None):
    """Query papers with filters"""
    try:
        sql = """
            SELECT 
                id,
                COALESCE(title, '(No title)') as title,
                COALESCE(journal, '(No journal)') as journal,
                year,
                COALESCE(doi, '') as doi,
                COALESCE(citation_count, 0) as citation_count
            FROM papers
            WHERE 1=1
        """
        params = []
        
        if tags:
            sql += " AND id IN (SELECT paper_id FROM tags WHERE tag = ANY(%s))"
            params.append([str(t) for t in tags])
            
        if year_start is not None and year_end is not None:
            sql += " AND year BETWEEN %s AND %s"
            params.extend([int(year_start), int(year_end)])
            
        if search:
            sql += " AND (title ILIKE %s OR abstract ILIKE %s)"
            search = f"%{search.strip()}%"
            params.extend([search, search])
            
        sql += " ORDER BY year DESC NULLS LAST, citation_count DESC NULLS LAST LIMIT 200"
        
        with _conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            
        df = pd.DataFrame(
            rows,
            columns=["ID", "Title", "Journal", "Year", "DOI", "Citations"]
        )
        
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Citations"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).astype(int)
        
        return df
        
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return pd.DataFrame(columns=["ID", "Title", "Journal", "Year", "DOI", "Citations"])

# Main app
st.title("Literature OS")
st.caption("Last updated via GitHub → HF sync ✅")

try:
    # Connect to database
    db = get_db()
    
    # Load filter options
    tags, years = get_metadata(db)
    
    # Filter section
    st.subheader("📋 Filter Papers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search = st.text_input(
            "Search titles & abstracts",
            help="Enter keywords to search in paper titles and abstracts"
        )
        
        year_range = st.slider(
            "Year range",
            min_value=2010,
            max_value=2025,
            value=(2015, 2025),
            help="Filter papers by publication year"
        )
    
    with col2:
        selected_tags = st.multiselect(
            "Filter by tags",
            options=sorted(tags),
            help="Select tags to filter papers"
        )
    
    # Query and display papers
    papers = get_papers(
        db,
        tags=selected_tags,
        year_start=year_range[0],
        year_end=year_range[1],
        search=search
    )
    
    st.subheader(f"📚 Results ({len(papers)} papers)")
    
    if papers.empty:
        st.info("No papers found matching your criteria.")
    else:
        papers['DOI'] = papers['DOI'].apply(lambda x: f'[{x}](https://doi.org/{x})' if x else '')
        
        st.dataframe(
            papers,
            use_container_width=True,
            column_config={
                "DOI": st.column_config.LinkColumn("DOI"),
                "Citations": st.column_config.NumberColumn("Citations", format="%d"),
                "Year": st.column_config.NumberColumn("Year", format="%d")
            },
            hide_index=True
        )
        
        csv = papers.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download as CSV",
            csv,
            "papers.csv",
            "text/csv",
            key='download-csv'
        )

except Exception as e:
    st.error(f"Application error: {str(e)}")

# Database functions
@st.cache_resource
def get_db():
    """Get database connection"""
    try:
        return psycopg.connect(
            os.environ["DB_READER_DSN"],
            application_name="literature-os-streamlit"
        )
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        raise

@st.cache_data(ttl=300)
def load_tags_years(_conn):
    """Load tags and years from database"""
    try:
        with _conn.cursor() as cur:
            # Get tags
            cur.execute("SELECT DISTINCT tag FROM tags ORDER BY tag")
            tags = [r[0] for r in cur.fetchall() if r[0]]
            
            # Get years
            cur.execute("""
                SELECT DISTINCT year 
                FROM papers 
                WHERE year IS NOT NULL AND year BETWEEN 1900 AND 2100
                ORDER BY year DESC
            """)
            years = [r[0] for r in cur.fetchall() if r[0]]
            
        return tags or [], years or list(range(2010, 2026))
    except Exception as e:
        st.error(f"Failed to load metadata: {str(e)}")
        return [], list(range(2010, 2026))

@st.cache_data(ttl=300)
def query_papers(_conn, tags=None, year_start=None, year_end=None, search=None):
    """Query papers with filters"""
    try:
        sql = """
            SELECT 
                id,
                COALESCE(title, '(No title)') as title,
                COALESCE(journal, '(No journal)') as journal,
                year,
                COALESCE(doi, '') as doi,
                COALESCE(citation_count, 0) as citation_count
            FROM papers
            WHERE 1=1
        """
        params = []
        
        # Apply filters
        if tags:
            sql += " AND id IN (SELECT paper_id FROM tags WHERE tag = ANY(%s))"
            params.append([str(t) for t in tags])
            
        if year_start and year_end:
            sql += " AND year BETWEEN %s AND %s"
            params.extend([year_start, year_end])
            
        if search:
            sql += " AND (title ILIKE %s OR abstract ILIKE %s)"
            search = f"%{search}%"
            params.extend([search, search])
            
        sql += " ORDER BY year DESC NULLS LAST, citation_count DESC NULLS LAST LIMIT 200"
        
        # Run query
        with _conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            
        # Convert to DataFrame
        df = pd.DataFrame(
            rows,
            columns=["ID", "Title", "Journal", "Year", "DOI", "Citations"]
        )
        
        # Fix types
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Citations"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).astype(int)
        
        return df
        
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return pd.DataFrame(columns=["ID", "Title", "Journal", "Year", "DOI", "Citations"])

# Main app
st.title("Literature OS")
st.caption("Last updated via GitHub → HF sync ✅")

# Database connection
try:
    db = get_db()
    
    # Load filter options
    tags, years = load_tags_years(db)
    
    # Filter section
    st.subheader("📋 Filter Papers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Search box
        search = st.text_input(
            "Search titles & abstracts",
            help="Enter keywords to search in paper titles and abstracts"
        )
        
        # Year range
        year_range = st.slider(
            "Year range",
            min_value=2010,
            max_value=2025,
            value=(2015, 2025),
            help="Filter papers by publication year"
        )
    
    with col2:
        # Tag selection
        selected_tags = st.multiselect(
            "Filter by tags",
            options=sorted(tags),
            help="Select tags to filter papers"
        )
    
    # Query papers with filters
    papers = query_papers(
        db,
        tags=selected_tags,
        year_start=year_range[0],
        year_end=year_range[1],
        search=search
    )
    
    # Display results
    st.subheader(f"📚 Results ({len(papers)} papers)")
    
    if papers.empty:
        st.info("No papers found matching your criteria.")
    else:
        # Add clickable DOI links
        papers['DOI'] = papers['DOI'].apply(lambda x: f'[{x}](https://doi.org/{x})' if x else '')
        
        # Display table
        st.dataframe(
            papers,
            use_container_width=True,
            column_config={
                "DOI": st.column_config.LinkColumn("DOI"),
                "Citations": st.column_config.NumberColumn("Citations", format="%d"),
                "Year": st.column_config.NumberColumn("Year", format="%d")
            },
            hide_index=True
        )
        
        # Download option
        csv = papers.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download as CSV",
            csv,
            "papers.csv",
            "text/csv",
            key='download-csv'
        )

except Exception as e:
    st.error(f"Application error: {str(e)}")


import streamlit as st
import pandas as pd
import psycopg

# Set up your page config
st.set_page_config(page_title="Literature OS", layout="wide")

st.caption("Last updated via GitHub → HF sync ✅")

# Initialize session state for error tracking
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

def log_error(e: Exception, context: str):
    """Log error with context"""
    error_msg = f"{context}: {str(e)}"
    st.error(error_msg)  # Use Streamlit's error display instead of logger
    st.session_state.error_log.append(error_msg)
    return error_msg

@st.cache_resource
def conn():
    try:
        return psycopg.connect(
            os.environ["DB_READER_DSN"],
            application_name="literature-os-streamlit"
        )
    except Exception as e:
        log_error(e, "Database connection failed")
        raise

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_tags_years(_c):
    try:
        with _c.cursor() as cur:
            # Load tags with error handling
            try:
                cur.execute("SELECT DISTINCT tag FROM tags ORDER BY tag")
                tags = [r[0] for r in cur.fetchall() if r[0]]  # Filter out None/empty tags
            except Exception as e:
                log_error(e, "Failed to load tags")
                tags = []

            # Load years with error handling
            try:
                cur.execute("""
                    SELECT DISTINCT year 
                    FROM papers 
                    WHERE year IS NOT NULL 
                        AND year BETWEEN 1900 AND 2100
                    ORDER BY year DESC
                """)
                years = [r[0] for r in cur.fetchall() if r[0]]  # Filter out None years
            except Exception as e:
                log_error(e, "Failed to load years")
                years = list(range(2010, 2026))  # Fallback years

        return tags or [], years or list(range(2010, 2026))
    except Exception as e:
        log_error(e, "Failed to load metadata")
        return [], list(range(2010, 2026))

def safe_escape_like(text):
    """Safely escape LIKE pattern wildcards"""
    if not isinstance(text, str):
        return text
    return text.replace('%', '\\%').replace('_', '\\_')

@st.cache_data(ttl=300)
def query(_c, tags=None, y0=None, y1=None, q=None):
    """Query papers with safe parameter handling"""
    try:
        # Start with base query
        sql = """
            SELECT 
                id,
                COALESCE(title, '(No title)') as title,
                COALESCE(journal, '(No journal)') as journal,
                year,
                COALESCE(doi, '') as doi,
                COALESCE(citation_count, 0) as citation_count
            FROM papers
            WHERE 1=1
        """
        params = []
        
        # Add filters one by one, with safe type checking
        if tags and isinstance(tags, (list, tuple)) and tags:
            sql += " AND id IN (SELECT paper_id FROM tags WHERE tag = ANY(%s))"
            params.append([str(t) for t in tags])  # Ensure all tags are strings
            
        if y0 is not None and y1 is not None:
            sql += " AND (year IS NOT NULL AND year BETWEEN %s AND %s)"
            params.extend([int(y0), int(y1)])
                
        if q and isinstance(q, str) and (q := q.strip()):  # Assignment expression
            sql += " AND (title ILIKE %s OR abstract ILIKE %s)"
            q = f"%{q}%"
            params.extend([q, q])
            
        # Always add ordering and limit
        sql += " ORDER BY year DESC NULLS LAST, citation_count DESC NULLS LAST LIMIT 200"
        
        # Debug information in expandable section
        with st.expander("🔍 Query Debug Info", expanded=False):
            st.code(sql)
            st.json({"parameters": str(params) if params else "None"})
        
        # Execute query with error handling
        with c.cursor() as cur:
            cur.execute(sql, params or None)
            rows = cur.fetchall()
            
        # Convert to DataFrame with proper types
        df = pd.DataFrame(rows, columns=["id","Title","Journal","Year","DOI","Citations"])
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Citations"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).astype(int)
        return df
        
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        st.code(f"SQL: {sql}\nParams: {params}")
        # Return empty DataFrame instead of raising
        return pd.DataFrame(columns=["id","Title","Journal","Year","DOI","Citations"])

# Main app
st.title("Literature OS - MVP")

# Simple database connection - no retries
try:
    c = conn()
    st.success("Connected to database ✅")
except Exception as e:
    st.error(f"Database connection failed: {str(e)}")
    st.stop()

# Load metadata with fallbacks
try:
    all_tags, years = load_tags_years(c)
except Exception as e:
    st.error(f"Error loading filters: {str(e)}")
    all_tags, years = [], list(range(2010, 2026))

# Filter section
st.subheader("📋 Filter Papers")

# Search box
search_query = st.text_input(
    "Search titles & abstracts",
    value="",
    help="Enter keywords to search in paper titles and abstracts"
)

# Year range (simplified)
year_range = st.slider(
    "Year range",
    min_value=2010,
    max_value=2025,
    value=(2015, 2025),
    help="Filter papers by publication year"
)
                
                # Update session state
                st.session_state.filters['year_range'] = year_range

            # Tags with categorized options
            with col2:
                prefix_map = {
                    "design:": "Research Design",
                    "domain:": "Domain",
                    "modality:": "Modality",
                    "dx:": "Diagnosis"
                }
                
                try:
                    # Group tags by prefix
                    tag_groups = {prefix: [] for prefix in prefix_map.keys()}
                    for tag in all_tags:
                        if not isinstance(tag, str):
                            continue
                        for prefix in prefix_map.keys():
                            if tag.startswith(prefix):
                                tag_groups[prefix].append(tag)
                                break
                    
                    # Create multiselect for each group that has tags
                    selected_tags = []
                    for prefix, label in prefix_map.items():
                        if tag_groups[prefix]:
                            group_key = f"tag_group_{prefix}"
                            group_tags = st.multiselect(
                                label,
                                options=sorted(tag_groups[prefix]),
                                default=st.session_state.filters.get('tags', []),
                                help=f"Select {label.lower()} tags to filter by",
                                key=group_key
                            )
                            selected_tags.extend(group_tags)
                    
                    # Update session state
                    st.session_state.filters['tags'] = selected_tags
                
                except Exception as e:
                    log_error(e, "Error in tag filtering")
                    selected_tags = []
                    
        except Exception as e:
            log_error(e, "Error in filter UI")
            search_query = ""
            year_range = (2015, 2025)
            selected_tags = []

    # Query and display results
    try:
        # Get filter values from session state
        yr0, yr1 = st.session_state.filters['year_range']
        df = query(
            c, 
            st.session_state.filters['tags'],
            yr0, yr1,
            st.session_state.filters['search']
        )
        
        # Results section
        st.subheader(f"📚 Results ({len(df)} papers)")
        
        if len(df) == 0:
            st.info("No papers found matching your criteria. Try adjusting the filters.")
        else:
            # Add clickable DOI links
            df['DOI'] = df['DOI'].apply(lambda x: f'[{x}](https://doi.org/{x})' if x else '')
            
            # Display the dataframe
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "DOI": st.column_config.LinkColumn("DOI"),
                    "Citations": st.column_config.NumberColumn("Citations", format="%d"),
                    "Year": st.column_config.NumberColumn("Year", format="%d")
                },
                hide_index=True
            )
            
            # Download button
            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download as CSV",
                    csv,
                    "papers.csv",
                    "text/csv",
                    key='download-csv'
                )

    except Exception as e:
        st.error("Error loading results")
        with st.expander("Error details"):
            st.code(str(e))
        
except Exception as e:
    st.error("Application Error")
    with st.expander("Error details"):
        st.code(str(e))
